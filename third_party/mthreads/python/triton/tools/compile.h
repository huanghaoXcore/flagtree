#ifndef TT_KERNEL_INCLUDES
#define TT_KERNEL_INCLUDES

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <{driver_include}>

#endif

void unload_{kernel_name}(void);
void load_{kernel_name}(void);
// tt-linker: {kernel_name}:{full_signature}:{algo_info}:{backend}
{result_ty} {_placeholder} {kernel_name}({stream_ty} stream, {signature});
