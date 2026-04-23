#include "TritonMTGPUToLLVM/MUSATranslation.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/LLVMTranslationInterface.h"
#include "triton/Tools/Sys/GetEnv.hpp"

#include "llvm/Support/Path.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>

namespace {

// split "-mtgpu-enable-const-calc=1 -mtgpu-alloc-shared-memory-from-zero=1" to
// "-mtgpu-enable-const-calc=1", "-mtgpu-alloc-shared-memory-from-zero=1"
static llvm::SmallVector<std::string, 8> splitByWhitespace(llvm::StringRef s) {
  llvm::SmallVector<std::string, 8> out;

  while (true) {
    s = s.ltrim();
    if (s.empty())
      break;

    size_t end = s.find_first_of(" \t\r\n");
    llvm::StringRef tok =
        (end == llvm::StringRef::npos) ? s : s.take_front(end);

    out.emplace_back(tok.str());

    if (end == llvm::StringRef::npos)
      break;
    s = s.drop_front(end);
  }
  return out;
}

std::string readStringFromEnv(const std::string &env_name,
                              const std::string &default_value) {
  std::string env_path = mlir::triton::tools::getStrEnv(env_name);
  return (!env_path.empty()) ? env_path : default_value;
}

void execute_llc(const std::string &mtcc_path,
                 llvm::ArrayRef<llvm::StringRef> args) {
  auto llc_program = llvm::sys::findProgramByName("llc", {mtcc_path});
  if (!llc_program) {
    llvm::errs() << "llc program not found in path: " << mtcc_path << "\n";
    assert("llc program not found in path!");
  }
  std::string err_msg;
  int ret = llvm::sys::ExecuteAndWait(*llc_program, args, std::nullopt, {}, 0,
                                      0, &err_msg);
  if (ret) {
    llvm::errs() << "llc execute fail: " << err_msg << "\n";
    assert("using llc to generate asm or obj failed!");
  }
}

void stripRangeAttributes(std::string &llStr) {
  size_t pos = 0;
  while ((pos = llStr.find("range(", pos)) != std::string::npos) {
    size_t start = pos;
    pos += 6; // skip "range("
    int depth = 1;
    while (pos < llStr.size() && depth > 0) {
      if (llStr[pos] == '(') {
        depth++;
      } else if (llStr[pos] == ')') {
        depth--;
      }
      pos++;
    }
    size_t end = pos;
    while (end < llStr.size() &&
           std::isspace(static_cast<unsigned char>(llStr[end]))) {
      end++;
    }
    if (end < llStr.size() && llStr[end] == '%') {
      llStr.erase(start, end - start);
      pos = start;
    }
  }
}

// convert latest llvm ir to mtcc compatible llvm ir.
// see llvm/docs/ReleaseNotes.rst
void convertLLVMIR(const std::string &filename) {
  // LLVM compatible. mtcc dependencies on llvm-14, convert llvm ir to mtcc
  // compatible format.
  auto make_llvm_compatible = [](std::string &ll_str) {
    // clang-format off
    std::vector<std::string> old_format = {
      "readnone",
      "readonly",
      "writeonly",
      "argmemonly",
      "argmemonly readonly",
      "argmemonly writeonly",
      "inaccessiblememonly",
      "inaccessiblememonly readonly",
      "inaccessiblememonly writeonly",
      "inaccessiblemem_or_argmemonly",
      "inaccessiblemem_or_argmemonly readonly",
      "inaccessiblemem_or_argmemonly writeonly"
    };
    std::vector<std::string> new_format = {
      "memory\\(none\\)",
      "memory\\(read\\)",
      "memory\\(write\\)",
      "memory\\(argmem: readwrite\\)",
      "memory\\(argmem: read\\)",
      "memory\\(argmem: write\\)",
      "memory\\(inaccessiblemem: readwrite\\)",
      "memory\\(inaccessiblemem: read\\)",
      "memory\\(inaccessiblemem: write\\)",
      "memory\\(argmem: readwrite, inaccessiblemem: readwrite\\)",
      "memory\\(argmem: read, inaccessiblemem: read\\)",
      "memory\\(argmem: write, inaccessiblemem: write\\)"
    };
    // clang-format on
    for (int i = 0; i < old_format.size(); ++i) {
      ll_str =
          std::regex_replace(ll_str, std::regex(new_format[i]), old_format[i]);
    }

    std::regex splat_regex(R"(<(\d+)\s+x\s+(\w+)> splat \(\2 ([^)]+)\))");
    std::smatch match;
    std::string result;
    std::string::const_iterator search_start(ll_str.cbegin());

    while (std::regex_search(search_start, ll_str.cend(), match, splat_regex)) {
      result.append(match.prefix().str());

      int count = std::stoi(match[1].str());
      std::string type = match[2].str();
      std::string value = match[3].str();

      result += "<" + match[1].str() + " x " + type + "> <";
      for (int i = 0; i < count; ++i) {
        result += type + " " + value;
        if (i != count - 1)
          result += ", ";
      }
      result += ">";

      search_start = match.suffix().first;
    }

    result.append(search_start, ll_str.cend());
    ll_str = result;
    static const std::regex kRe(R"(\bicmp\s+samesign\b)");
    ll_str = std::regex_replace(ll_str, kRe, "icmp");
    {
      std::regex line_regex(R"(^(?=.*<\s*1\s*x\s*(\w+)\s*>)(.*?))");
      std::regex splat_1_regex; // 动态按行内捕获到的类型构造

      std::string out;
      out.reserve(ll_str.size());
      std::istringstream iss(ll_str);
      std::string line;

      while (std::getline(iss, line)) {
        std::smatch lm;
        if (std::regex_search(line, lm, line_regex)) {
          // lm[1] 是向量元素类型 T
          const std::string ty = lm[1].str();
          // 构造 "splat (T V)"
          // 的匹配：splat后是任意空格、T、空格、直到右括号之前的值
          const std::string pat =
              R"(splat\s*\(\s*)" + ty + R"(\s+([^)]+)\s*\))";
          splat_1_regex.assign(pat);

          line = std::regex_replace(line, splat_1_regex, "<" + ty + " $1>");
        }
        out.append(line);
        out.push_back('\n');
      }
      ll_str = std::move(out);
    }

    {
      static const std::regex vec_regex(
          R"(<\s*(\d+)\s*x\s*([A-Za-z0-9_.]+)\s*>)");
      static const std::regex bare_splat_regex(
          R"(splat\s*\(\s*([A-Za-z0-9_.]+)\s+([^)]+)\s*\))");

      std::string out;
      out.reserve(ll_str.size());
      std::istringstream iss(ll_str);
      std::string line;

      while (std::getline(iss, line)) {
        size_t search_pos = 0;
        while (search_pos < line.size()) {
          std::string remaining = line.substr(search_pos);
          std::smatch splat_match;
          if (!std::regex_search(remaining, splat_match, bare_splat_regex))
            break;
          size_t rel_pos = splat_match.position();
          const std::string type = splat_match[1].str();
          const std::string value = splat_match[2].str();

          std::string prefix = line.substr(0, search_pos + rel_pos);
          std::smatch vec_match;
          std::string::const_iterator vec_search_start = prefix.cbegin();
          int lane_count = -1;
          while (std::regex_search(vec_search_start, prefix.cend(), vec_match,
                                   vec_regex)) {
            if (vec_match[2].str() == type) {
              lane_count = std::stoi(vec_match[1].str());
            }
            vec_search_start = vec_match.suffix().first;
          }

          if (lane_count <= 0) {
            search_pos += rel_pos + splat_match.length();
            continue;
          }

          const std::string lane_str = std::to_string(lane_count);
          const std::string vec_ty = "<" + lane_str + " x " + type + ">";
          const std::string mask_ty = "<" + lane_str + " x i32>";
          std::string insertelement_expr = "insertelement (" + vec_ty +
                                           " undef, " + type + " " + value +
                                           ", i32 0)";
          std::string replacement = "shufflevector (" + vec_ty + " " +
                                    insertelement_expr + ", " + vec_ty +
                                    " undef, " + mask_ty + " zeroinitializer)";

          line.replace(search_pos + rel_pos, splat_match.length(), replacement);
          search_pos += rel_pos + replacement.size();
        }
        out.append(line);
        out.push_back('\n');
      }
      ll_str = std::move(out);
    }

    stripRangeAttributes(ll_str);
  };

  // convert latest llvm ir to mtcc compatible llvm ir.
  std::ifstream is(filename);
  std::string ll_str((std::istreambuf_iterator<char>(is)),
                     std::istreambuf_iterator<char>());
  is.close();
  make_llvm_compatible(ll_str);

  // save the mtcc compatible llvm ir to ll file.
  std::ofstream os(filename);
  os << ll_str;
  os.close();

  if (mlir::triton::tools::getBoolEnv("MUSA_LLVMIR_ENABLE_DUMP")) {
    std::cout << "// -----// MUSA LLVMIR Dump //----- //\n"
              << ll_str << std::endl;
  }
}

std::string generate_muasm(const llvm::Module &llvmModule,
                           const std::string &opt_option, const int capability,
                           const int version, std::string &ll_file_name) {
  std::string function_name;
  std::string ll_file;
  std::string asm_file;

  llvm::SmallString<128> kernel;
  llvm::sys::fs::createTemporaryFile("mt_triton_kernel", /*suffix*/ "ll",
                                     kernel);
  ll_file = llvm::StringRef(kernel).str();
  ll_file_name = ll_file;
  llvm::sys::path::replace_extension(kernel, "s");
  asm_file = llvm::StringRef(kernel).str();

  std::error_code ec;
  llvm::raw_fd_ostream os(ll_file, ec, llvm::sys::fs::OF_None);
  llvmModule.print(os, nullptr);
  os.close();

  // get the name of mtgpu kernel.
  for (auto &F : llvmModule.getFunctionList()) {
    if (!F.isDeclaration() &&
        F.getCallingConv() == llvm::CallingConv::MTGPU_KERNEL) {
      function_name = F.getName().str();
      break;
    }
  }

  // convert latest llvm ir to mtcc compatible llvm ir.
  convertLLVMIR(ll_file);

  std::string replace_llfile = readStringFromEnv("MTGPU_REPLACE_LL", "");
  if (std::filesystem::exists(replace_llfile)) {
    printf(" *** replace llir %s -> %s\n", ll_file.c_str(),
           replace_llfile.c_str());
    ll_file = replace_llfile;
    ll_file_name = ll_file;
  }

  // because mtcc's building script has an option --disable_asm (default:
  // False), which can control mtcc's llc whether can support -filetype=asm or
  // not. so here we use an ENV: MTCC_ENABLE_ASM_BIN_PATH to indicate that this
  // path's llc can support -filetype=asm.
  //
  // by default, we use /usr/local/musa/bin/llc, which can't support
  // -filetype=asm, so we return the name of mtgpu kernel. otherwise, if we set
  // the ENV: MTCC_ENABLE_ASM_BIN_PATH, we will return the generated asm code.
  std::string mtcc_enable_asm_bin_path =
      readStringFromEnv("MTCC_ENABLE_ASM_BIN_PATH", "");

  if (!mtcc_enable_asm_bin_path.empty()) {
    // set ENV: MTCC_ENABLE_ASM_BIN_PATH, so return the generated asm code.
    // llc out.ll -march=mtgpu -O2 -filetype=asm -o out.asm
    std::string assign_subtarget = "-mcpu=mp_" + std::to_string(capability);
    llvm::SmallVector<llvm::StringRef> args{
        llvm::StringRef("llc"),
        llvm::StringRef(ll_file),
        llvm::StringRef("-march=mtgpu"),
        llvm::StringRef(assign_subtarget),
        llvm::StringRef("--opaque-pointers"),
        llvm::StringRef("-filetype=asm"),
        llvm::StringRef("-o"),
        llvm::StringRef(asm_file),
        llvm::StringRef("-O2")};

    llvm::SmallVector<std::string, 8> opt_tokens =
        splitByWhitespace(opt_option);
    args.reserve(args.size() + opt_tokens.size());
    for (auto &t : opt_tokens)
      args.push_back(t);

    // use the mtcc_enable_asm_bin_path's llc to generate asm code.
    execute_llc(mtcc_enable_asm_bin_path, args);

    // get the muasm code.
    std::ifstream is(asm_file);
    std::string muasm((std::istreambuf_iterator<char>(is)),
                      std::istreambuf_iterator<char>());
    is.close();

    if (mlir::triton::tools::getBoolEnv("MUASM_ENABLE_DUMP")) {
      std::cout << "// -----// MUASM Dump //----- //\n" << muasm << std::endl;
    }

    return muasm;
  } else {
    // by default, /usr/local/musa/bin/llc can't support -filetype=asm,
    // so return the name of mtgpu kernel.
    return ".globl\t" + function_name;
  }
}

std::string generate_mubin(const std::string &ll_file_name,
                           const std::string &opt_option, const int capability,
                           const int version) {
  int pos = ll_file_name.find_last_of('.');
  std::string obj_file = ll_file_name.substr(0, pos + 1) + "o";
  std::string lld_obj_file = ll_file_name.substr(0, pos + 1) + "lld.o.mubin";

  // llc out.ll -march=mtgpu -O2 -filetype=obj -o out.o
  std::string assign_subtarget = "-mcpu=mp_" + std::to_string(capability);
  llvm::SmallVector<llvm::StringRef> args{llvm::StringRef("llc"),
                                          llvm::StringRef(ll_file_name),
                                          llvm::StringRef("-march=mtgpu"),
                                          llvm::StringRef(assign_subtarget),
                                          llvm::StringRef("--opaque-pointers"),
                                          llvm::StringRef("-filetype=obj"),
                                          llvm::StringRef("-o"),
                                          llvm::StringRef(obj_file),
                                          llvm::StringRef("-O2")};

  llvm::SmallVector<std::string, 8> opt_tokens = splitByWhitespace(opt_option);
  args.reserve(args.size() + opt_tokens.size());
  for (auto &t : opt_tokens)
    args.push_back(t);

  // by default, we use the /usr/local/musa/bin/llc.
  // if we set the ENV: MTCC_ENABLE_ASM_BIN_PATH,
  // we should keep using the same llc tool with function: generate_muasm
  std::string mtcc_path =
      readStringFromEnv("MTCC_BIN_PATH", "/usr/local/musa/bin");
  std::string mtcc_enable_asm_bin_path =
      readStringFromEnv("MTCC_ENABLE_ASM_BIN_PATH", "");

  if (!mtcc_enable_asm_bin_path.empty()) {
    execute_llc(mtcc_enable_asm_bin_path, args);
  } else {
    // TODO: pre-install MTCC in docker or build bin in third_party
    execute_llc(mtcc_path, args);
  }

  // lld -flavor gnu -shared %bin -o %obj
  // clang-format off
  llvm::SmallVector<llvm::StringRef> lld_args{
    llvm::StringRef("ld.lld"),
    llvm::StringRef("-flavor"),
    llvm::StringRef("gnu"),
    llvm::StringRef("-shared"),
    llvm::StringRef(obj_file),
    llvm::StringRef("-o"),
    llvm::StringRef(lld_obj_file)
  };
  // clang-format on
  auto lld_program = llvm::sys::findProgramByName("ld.lld", {mtcc_path});
  if (!lld_program) {
    llvm::errs() << "lld program not found in path: " << mtcc_path << "\n";
    assert("using llc to generate obj failed!");
  }

  std::string err_msg;
  int lld_ret = llvm::sys::ExecuteAndWait(*lld_program, lld_args, std::nullopt,
                                          {}, 0, 0, &err_msg);
  if (lld_ret) {
    llvm::errs() << "lld execute fail: " << err_msg << "\n";
    assert("using llc to generate obj failed!");
  }

  return lld_obj_file;
}

std::tuple<std::string, std::string>
llir_to_muasm_and_mubin(llvm::Module *module, const std::string &opt_option,
                        int capability, int version) {
  std::string ll_file_name;
  auto muasm =
      generate_muasm(*module, opt_option, capability, version, ll_file_name);
  auto mubin_path =
      generate_mubin(ll_file_name, opt_option, capability, version);

  std::string replace_mubin_path = readStringFromEnv("MTGPU_REPLACE_BIN", "");
  if (std::filesystem::exists(replace_mubin_path)) {
    printf(" *** replace mubin_path %s -> %s\n", mubin_path.c_str(),
           replace_mubin_path.c_str());
    mubin_path = replace_mubin_path;
  }

  return std::make_tuple(muasm, mubin_path);
}

} // namespace

namespace mlir::triton {

std::tuple<std::string, std::string>
translateLLVMIRToMUBIN(llvm::Module &module, const std::string &opt_option,
                       int capability, int version) {
  auto muCode =
      llir_to_muasm_and_mubin(&module, opt_option, capability, version);
  return muCode;
}

} // namespace mlir::triton
