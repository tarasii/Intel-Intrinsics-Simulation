#ifndef __TYPES_H__
#define __TYPES_H__

#include <stdint.h>

#ifdef WIN32
#define __attribute(...)
#define __attribute__(...)
#else
typedef long long __int64;
typedef uint16_t __int16;
#define max(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })
#define min(a,b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })
#endif



#endif /*__TYPES_H__ */
