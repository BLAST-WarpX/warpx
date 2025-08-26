#ifndef DOXYGRAPH_H
#define DOXYGRAPH_H

#ifdef __DOXYGEN__
#  define DOXY_GRAPH( ... ) __VA_ARGS__
#else
#  define DOXY_GRAPH( ... )
#endif

#endif // DOXYGRAPH_H
