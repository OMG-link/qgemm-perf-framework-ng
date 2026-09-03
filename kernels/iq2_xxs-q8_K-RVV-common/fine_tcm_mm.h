#ifndef FINE_TCM_MM_H
#define FINE_TCM_MM_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

int fine_mm_init(size_t max_tcm_size);
void fine_mm_deinit(void);

__attribute__((malloc)) void *fine_mm_malloc(size_t size);
__attribute__((malloc)) void *fine_mm_calloc(size_t nmemb, size_t size);
void *fine_mm_realloc(void *ptr, size_t size);
void fine_mm_free(void *ptr);

int fine_mm_is_inited(void);

#ifdef __cplusplus
}
#endif

#endif