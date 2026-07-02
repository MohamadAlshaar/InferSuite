/* LD_PRELOAD shim: force every CUDA event to be created with the BLOCKING_SYNC flag, so
 * cudaEventSynchronize / cuEventSynchronize put the host thread to SLEEP instead of busy-waiting.
 * (cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync) does NOT affect cudaEventSynchronize — that is
 *  governed by the per-event flag, which is what we override here.)
 * cudaEventBlockingSync = 0x01 (runtime);  CU_EVENT_BLOCKING_SYNC = 0x01 (driver). */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>

#define BLOCKING 0x01u
static int announced = 0;
static void announce(const char* who){ if(!announced){ announced=1;
    fprintf(stderr, "[evblock] forcing BLOCKING_SYNC on CUDA events (via %s)\n", who); } }

typedef int (*cewf_t)(void*, unsigned int);
typedef int (*cec_t)(void*);

/* runtime: cudaEventCreateWithFlags(event, flags) */
int cudaEventCreateWithFlags(void* e, unsigned int flags){
    static cewf_t real=0; if(!real) real=(cewf_t)dlsym(RTLD_NEXT,"cudaEventCreateWithFlags");
    announce("cudaEventCreateWithFlags");
    return real(e, flags | BLOCKING);
}
/* runtime: cudaEventCreate(event) == default flags */
int cudaEventCreate(void* e){
    static cewf_t real=0; if(!real) real=(cewf_t)dlsym(RTLD_NEXT,"cudaEventCreateWithFlags");
    announce("cudaEventCreate");
    return real(e, BLOCKING);
}
/* driver: cuEventCreate(phEvent, Flags) */
int cuEventCreate(void* e, unsigned int flags){
    static cewf_t real=0; if(!real) real=(cewf_t)dlsym(RTLD_NEXT,"cuEventCreate");
    announce("cuEventCreate");
    return real(e, flags | BLOCKING);
}
