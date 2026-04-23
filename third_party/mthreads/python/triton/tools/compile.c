/* clang-format off */
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <string.h>
#include <{driver_include}>


// helpers to check for driver errors
#define DRIVER_CHECK(ans) {{\
    gpuAssert((ans), __FILE__, __LINE__);\
  }}\

static inline void gpuAssert({result_ty} code, const char *file, int line) {{
  if (code != {success_value}) {{
    const char *prefix = "Triton Error [{error_prefix}]: ";
    const char *str;
    {get_error_string_fn}(code, &str);
    char err[1024] = {{0}};
    strcat(err, prefix);
    strcat(err, str);
    printf("%s\\n", err);
    exit(code);
  }}
}}

// globals
#define BINARY_NAME {kernel_name}_{binary_suffix}
{module_ty} {kernel_name}_mod = NULL;
{function_ty} {kernel_name}_func = NULL;
unsigned char BINARY_NAME[{bin_size}] = {{ {bin_data} }};


void unload_{kernel_name}(void) {{
    if ({kernel_name}_mod != NULL) {{
      DRIVER_CHECK({module_unload_fn}({kernel_name}_mod));
      {kernel_name}_mod = NULL;
      {kernel_name}_func = NULL;
    }}
}}

// TODO: some code duplication with runtime backend loaders.
void load_{kernel_name}() {{
    void *bin = (void *)&BINARY_NAME;
{load_preamble}    DRIVER_CHECK({module_load_data_fn}(&{kernel_name}_mod, bin));
    DRIVER_CHECK({module_get_function_fn}(&{kernel_name}_func, {kernel_name}_mod, "{triton_kernel_name}"));
{post_load_setup}}}

/*
{kernel_docstring}
*/
{result_ty} {kernel_name}({stream_ty} stream, {signature}) {{
    if ({kernel_name}_func == NULL)
       load_{kernel_name}();
    unsigned int gX = {gridX};
    unsigned int gY = {gridY};
    unsigned int gZ = {gridZ};
    void *args[{num_args}] = {{ {arg_pointers} }};
    if(gX * gY * gZ > 0)
      return {launch_fn}({kernel_name}_func, gX, gY, gZ, {block_dim_x}, 1, 1, {shared}, stream, args, NULL);
    return {success_value};
}}
