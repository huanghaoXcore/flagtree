from triton.backends.compiler import BaseBackend, GPUTarget
from triton._C.libtriton import ir, passes, llvm, mthreads

from dataclasses import dataclass
import functools
from typing import Any, Dict, Tuple, Optional
from types import ModuleType
import hashlib
import tempfile
import signal
import os
import subprocess
from pathlib import Path
import shutil
import warnings


def get_kernel_name(src: str, pattern: str) -> str:
    assert src
    for line in src.split("\n"):
        line = line.strip()
        if line.startswith(pattern):
            return line.split()[-1]


@functools.lru_cache()
def get_musa_version():
    version = subprocess.check_output(["/usr/local/musa/bin/musa_toolkits_version"]).decode("utf-8")
    return version


@functools.lru_cache(None)
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


_SQMMA_OP_NAMES = {
    "triton_mthreads_gpu.squad_dot",
    "triton_mthreads_gpu.squad_dot_wait",
}


def _module_uses_sqmma(mod) -> bool:
    uses_sqmma = False

    def walk_fn(op):
        nonlocal uses_sqmma
        if uses_sqmma:
            return
        if op.get_name() in _SQMMA_OP_NAMES:
            uses_sqmma = True

    mod.walk(walk_fn)
    return uses_sqmma


@dataclass(frozen=True)
class MUSAOptions:
    num_warps: int = 4
    num_ctas: int = 1
    num_stages: int = 1
    num_buffers_warp_spec: int = 0
    num_consumer_groups: int = 0
    reg_dec_producer: int = 0
    reg_inc_consumer: int = 0
    maxnreg: Optional[int] = None
    cluster_dims: tuple = (1, 1, 1)
    capability: int = None
    enable_fp_fusion: bool = True
    supported_fp8_dtypes: Tuple[str] = ("fp8e5", "fp8e4nv")
    deprecated_fp8_dtypes: Tuple[str] = ()
    default_dot_input_precision: str = "tf32"
    allowed_dot_input_precisions: Tuple[str] = ("tf32", "tf32x3", "ieee")
    max_num_imprecise_acc_default: int = 0
    extern_libs: dict = None
    debug: bool = False
    backend_name: str = "musa"
    en_wmma: bool = False
    sanitize_overflow: bool = True
    en_backend_opt: bool = False

    def __post_init__(self):
        extern_libs = {} if self.extern_libs is None else dict(self.extern_libs)
        if not extern_libs.get("libdevice", None):
            if self.capability >= 31:
                default_libdir = "/usr/local/musa/mtgpu/bitcode/libdevice.31.bc"
            else:
                default_libdir = "/usr/local/musa/mtgpu/bitcode/libdevice.bc"
            # here we add an new ENV: MUSA_LIBDEVICE_PATH for MUSA,
            # which represents the path of libdevice.bc
            musa_env_path = os.environ.get("MUSA_LIBDEVICE_PATH", default_libdir)
            extern_libs["libdevice"] = musa_env_path
        object.__setattr__(self, "extern_libs", tuple(extern_libs.items()))
        assert (self.num_warps > 0 and (self.num_warps & (self.num_warps - 1)) == 0), "num_warps must be a power of 2"

    def hash(self):
        hash_dict = dict(self.__dict__)
        hash_dict["extern_libs"] = tuple((k, file_hash(v)) for k, v in sorted(hash_dict["extern_libs"]))
        key = "_".join([f"{name}-{val}" for name, val in sorted(hash_dict.items())])
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    def parse_options(self, opts) -> Any:
        opts["capability"] = self.capability
        opts["allow_fp8e4nv"] = self.capability >= 31
        opts["en_wmma"] = (opts.get("en_wmma") if opts.get("en_wmma") is not None else
                           (os.getenv("ENABLE_MUSA_MMA", "false").lower() != "false"))
        opts["en_backend_opt"] = False if opts.get("en_backend_opt") is None else opts["en_backend_opt"]
        args = {k: opts[k] for k in MUSAOptions.__dataclass_fields__.keys() if k in opts}
        return MUSAOptions(**args)


class MUSABackend(BaseBackend):

    @staticmethod
    def supports_target(target: GPUTarget):
        return target.backend == "musa"

    def __init__(self, target: GPUTarget) -> None:
        super().__init__(target)
        self.capability = target.arch
        self.warp_size = target.warp_size
        assert isinstance(self.capability, int)
        self.binary_ext = "mubin"

    def parse_options(self, opts) -> Any:
        opts["capability"] = self.capability
        opts["allow_fp8e4nv"] = self.capability >= 31
        opts["en_wmma"] = (opts.get("en_wmma") if opts.get("en_wmma") is not None else
                           (os.getenv("ENABLE_MUSA_MMA", "false").lower() != "false"))
        args = {k: opts[k] for k in MUSAOptions.__dataclass_fields__.keys() if k in opts}
        return MUSAOptions(**args)

    def pack_metadata(self, metadata):
        return (
            metadata.num_warps,
            metadata.num_ctas,
            metadata.shared,
            metadata.cluster_dims[0],
            metadata.cluster_dims[1],
            metadata.cluster_dims[2],
        )

    def get_codegen_implementation(self):
        import triton.language.extra.musa as musa

        def _min_dot_size(lhs_type, rhs_type):
            return (1, 1, 1)

        codegen_fns = {
            "convert_custom_types": None,
            "min_dot_size": _min_dot_size,
            "max_num_imprecise_acc_default": lambda: 0,
        }
        return codegen_fns

    def get_module_map(self) -> Dict[str, ModuleType]:
        from triton.language.extra.musa import libdevice
        return {"triton.language.extra.libdevice": libdevice}

    def load_dialects(self, ctx):
        mthreads.load_dialects(ctx)

    @staticmethod
    def make_ttir(mod, metadata, opt):
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.common.add_inliner(pm)
        passes.ttir.add_rewrite_tensor_pointer(pm)
        passes.ttir.add_combine(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_reorder_broadcast(pm)
        passes.common.add_cse(pm)
        passes.common.add_licm(pm)
        passes.common.add_symbol_dce(pm)
        passes.ttir.add_loop_unroll(pm)
        pm.run(mod)
        return mod

    @staticmethod
    def make_ttgir(mod, metadata, opt, capability, warp_size):
        disable_sqmma = os.environ.get("DISABLE_SQMMA")
        enable_sqmma = disable_sqmma is None
        # TTIR -> TTGIR
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.ttir.add_convert_to_ttgpuir(pm, f"musa:{capability}", opt.num_warps, warp_size, opt.num_ctas)
        # optimize TTGIR
        passes.ttgpuir.add_coalesce(pm)
        if enable_sqmma:
            passes.ttgpuir.add_optimize_thread_locality(pm)
            mthreads.passes.ttgpuir.add_accelerate_sqmma(pm)
        else:
            passes.ttgpuir.add_remove_layout_conversions(pm)
        mthreads.passes.ttgpuir.add_accelerate_wmma(pm, opt.en_wmma)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 31)
        passes.common.add_cse(pm)
        if enable_sqmma:
            passes.ttgpuir.add_optimize_accumulator_init(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        mthreads.passes.ttmtgpuir.add_mtgpu_pipeline(pm, opt.num_stages)
        passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 31)
        passes.ttgpuir.add_remove_layout_conversions(pm)
        passes.ttgpuir.add_reduce_data_duplication(pm)
        passes.ttgpuir.add_reorder_instructions(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        mthreads.passes.ttmtgpuir.add_tme_lowering(pm)
        passes.common.add_canonicalizer(pm)
        if capability == 31:
            mthreads.passes.ttgpuir.add_mark_inplace_loads(pm)
        pm.run(mod)
        metadata["uses_sqmma"] = _module_uses_sqmma(mod)
        return mod

    @staticmethod
    def make_llir(src, metadata, options, capability):
        # warp-specialization mutates num_warps
        num_warp_groups = src.get_int_attr("triton_gpu.num-warp-groups-per-cta")
        if num_warp_groups is not None:
            metadata["num_warps"] *= num_warp_groups
        mod = src
        # TritonGPU -> LLVM-IR (MLIR)
        pm = ir.pass_manager(mod.context)
        pm.enable_debug()
        passes.convert.add_scf_to_cf(pm)
        passes.convert.add_index_to_llvmir(pm)
        passes.ttgpuir.add_allocate_shared_memory(pm)
        mthreads.passes.ttgpuir.add_to_llvmir(pm, capability)
        # NOTE: ttgpu -> mtgpu -> llvm
        mthreads.passes.ttmtgpuir.add_mtgpu_to_llvm(pm)
        passes.convert.add_arith_to_llvmir(pm)
        passes.common.add_canonicalizer(pm)
        passes.common.add_cse(pm)
        passes.common.add_symbol_dce(pm)
        if os.environ.get("TRITON_DISABLE_LINE_INFO", "0") == "0":
            passes.llvmir.add_di_scope(pm)

        mthreads.passes.ttgpuir.add_mtgpu_builtin_func_to_llvmir(pm)
        if capability == 31:
            mthreads.passes.ttgpuir.add_inplace_load_to_llvm(pm)
        pm.run(mod)
        # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
        llvm.init_targets()
        context = llvm.context()

        llvm_mod = llvm.to_module(mod, context)
        mthreads.attach_datalayout(llvm_mod)

        if options.extern_libs:
            paths = [path for (name, path) in options.extern_libs]
            llvm.link_extern_libs(llvm_mod, paths)

        llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)

        # Get some metadata
        metadata["shared"] = src.get_int_attr("triton_gpu.shared")
        ret = str(llvm_mod)
        del llvm_mod
        del context
        return ret

    @staticmethod
    def _build_llc_options(uses_sqmma: bool, enable_backend_opt: bool) -> list:
        # Flag sets keyed by (uses_sqmma, enable_backend_opt).
        # Note: the (True, True) set uses -mtgpu-opt-level which subsumes -mtgpu-enable-const-calc.
        # Available flags:
        #   -mtgpu-if-convert                    use p0 to convert branch
        #   -mtgpu-combine-instr-with-burst      vectorize in llvm ir
        #   -mtgpu-combine-fop-instr             vectorize after RA
        #   -mtgpu-tiny-offset-hint              assume offset is 32 bit
        #   -mtgpu-enable-const-calc             move uniform instructions to a separate kernel
        #   -mtgpu-alloc-shared-memory-from-zero don't reserve shared memory
        #   -mtgpu-opt-level                     bundled optimisation set (equiv. -Od3)
        options = {
            (False, False): [
                "-mtgpu-enable-const-calc=1",
            ],
            (True, False): [
                "-mtgpu-enable-const-calc=1",
                "-mtgpu-alloc-shared-memory-from-zero=1",
            ],
            (False, True): [
                "-mtgpu-enable-const-calc=1",
                "-mtgpu-tiny-offset-hint=1",
                "-mtgpu-combine-instr-with-burst=1",
                "-mtgpu-combine-fop-instr=1",
            ],
            (True, True): [
                "-mtgpu-opt-level=1",
                "-mtgpu-combine-instr-with-burst=1",
                "-mtgpu-combine-fop-instr=1",
                "-misched=mtgpu-max-ilp",
            ],
        }
        return options[(uses_sqmma, enable_backend_opt)]

    @staticmethod
    def _dump_if_enabled(env_var: str, header: str, content: str) -> None:
        if os.environ.get(env_var, "0") == "1":
            print(header)
            print(content)

    @staticmethod
    def _maybe_replace_ir(src: str) -> str:
        """DEBUG: replace LLVM IR with replace.llir if MUSA_REPLACE_IR=1."""
        if os.environ.get("MUSA_REPLACE_IR", "0") != "1":
            return src
        file_path = '/usr/local/musa/mtgpu/bitcode/replace.llir'
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                src = f.read()
            MUSABackend._dump_if_enabled("MUSA_REPLACE_IR", "// -----// After replace, new ir is:", src)
        except FileNotFoundError:
            print(f"error: {file_path} does not exist, replace ir failed")
        except PermissionError:
            print(f"error: no permission to read {file_path}, replace ir failed")
        except Exception as e:
            print(f"error: failed to read {file_path}: {e}, replace ir failed")
        return src

    @staticmethod
    def _maybe_save_mubin(ret) -> None:
        """Save compiled binary to MUBIN_SAVE_PATH if set."""
        mubin_save_path = os.environ.get("MUBIN_SAVE_PATH", "")
        if mubin_save_path:
            shutil.copy2(ret[1], os.path.join(mubin_save_path, "test.out"))

    @staticmethod
    def make_mubin(src, metadata, opt, capability):
        """
        Translate TritonGPU module to MUSA binary code.
        """
        MUSABackend._dump_if_enabled("MUSA_ENABLE_DUMP_LLVM", "// -----// LLVM IR", src)

        uses_sqmma = bool(metadata.get("uses_sqmma"))
        enable_backend_opt = os.environ.get("MUSA_ENABLE_LLC_OPT", "0") == "1" or opt.en_backend_opt
        opt_option = " ".join(MUSABackend._build_llc_options(uses_sqmma, enable_backend_opt))

        src = MUSABackend._maybe_replace_ir(src)
        ret = mthreads.translate_llvmir_to_mubin(src, opt_option, capability, 0)
        MUSABackend._dump_if_enabled("MUSA_ASM_ENABLE_DUMP", "// -----// MTGPU ASM", ret[0])
        MUSABackend._maybe_save_mubin(ret)
        metadata["name"] = get_kernel_name(ret[0], pattern=".globl")
        return ret

    def add_stages(self, stages, options):
        stages["ttir"] = lambda src, metadata: self.make_ttir(src, metadata, options)
        stages["ttgir"] = lambda src, metadata: self.make_ttgir(src, metadata, options, self.capability, self.warp_size)
        stages["llir"] = lambda src, metadata: self.make_llir(src, metadata, options, self.capability)
        stages["mubin"] = lambda src, metadata: self.make_mubin(src, metadata, options, self.capability)

    @functools.lru_cache()
    def hash(self):
        version = get_musa_version()
        return f"{version}-{self.capability}"
