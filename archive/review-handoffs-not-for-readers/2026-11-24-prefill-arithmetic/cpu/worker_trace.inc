/* Isolated decode observer. One synchronous CPU graph at a time. No I/O between caller clocks. */
#include <sched.h>
#include <stdatomic.h>
#include <unistd.h>
#define WB_THREADS 48
#define WB_NODES 400
#define WB_EXPERTS 6

struct wb_expert {
    int id, inputs, compact, rows, chunks, tiled;
    uint64_t start, end;
};
struct wb_record {
    uint64_t start, prep_in, prep_out, finish, released;
    int graph, index, ith, nth, tid, cpu, end_cpu, op, nrows, nin, qtype, sync, nexperts;
    char name[GGML_MAX_NAME];
    struct wb_expert e[WB_EXPERTS];
};
struct wb_stream {
    int count;
    struct wb_record rec[WB_NODES];
} __attribute__((aligned(64)));
static struct wb_stream wb_buffers[WB_THREADS];
static _Atomic int wb_on, wb_graph, wb_graph_busy;
static _Thread_local int wb_tid;
static _Thread_local struct wb_record *wb_current;
static inline uint64_t wb_clock(void) {
    struct timespec t;
    GGML_ASSERT(clock_gettime(CLOCK_MONOTONIC, &t) == 0);
    return (uint64_t)t.tv_sec * UINT64_C(1000000000) + t.tv_nsec;
}

GGML_API void xeon_worker_trace_begin(void);
GGML_API int xeon_worker_trace_end(const char *name);
void xeon_worker_trace_begin(void) {
    GGML_ASSERT(!atomic_load(&wb_on) && !atomic_load(&wb_graph_busy));
    for (int t=0; t<WB_THREADS; t++) wb_buffers[t].count=0;
    atomic_store(&wb_graph, 0);
    atomic_store(&wb_on, 1);
}
static inline struct wb_record *wb_node(int graph, int index, int ith, int nth, const struct ggml_tensor *node) {
    if (!graph) return NULL;
    GGML_ASSERT(ith>=0 && ith<WB_THREADS && nth<=WB_THREADS);
    struct wb_stream *s=&wb_buffers[ith];
    GGML_ASSERT(s->count<WB_NODES);
    struct wb_record *r=&s->rec[s->count++];
    memset(r,0,sizeof(*r));
    if (!wb_tid) wb_tid=(int)syscall(SYS_gettid);
    r->graph=graph; r->index=index; r->ith=ith; r->nth=nth; r->tid=wb_tid;
    r->cpu=sched_getcpu(); r->qtype=-1;
    if (node) {
        memcpy(r->name,node->name,sizeof(r->name));
        r->op=node->op;
        if (node->op==GGML_OP_MUL_MAT_ID) {
            r->qtype=node->src[0]->type; r->nrows=(int)node->src[0]->ne[1]; r->nin=(int)node->src[2]->ne[1];
        }
    } else {
        strcpy(r->name,"__graph_end__"); r->op=-1;
    }
    wb_current=r;
    r->start=wb_clock();
    return r;
}

int xeon_worker_trace_end(const char *name) {
    GGML_ASSERT(atomic_load(&wb_on) && !atomic_load(&wb_graph_busy));
    atomic_store(&wb_on,0);
    FILE *f=fopen(name,"wx");
    if (!f) return -1;
    fprintf(f,"graph\tindex\tname\top\tqtype\tnrows\tnin\tith\tnth\ttid\tcpu\tend_cpu\tstart\tprep_in\tprep_out\tfinish\treleased\tsync\texperts\n");
    for (int t=0; t<WB_THREADS;t++) {
        for (int n=0;n<wb_buffers[t].count;n++) {
            const struct wb_record *r=&wb_buffers[t].rec[n];
            fprintf(f,"%d\t%d\t%s\t%s\t%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%"PRIu64"\t%d\t",
                r->graph,r->index,r->name,r->op<0?"graph_end":ggml_op_name(r->op),r->qtype<0?"-":ggml_type_name(r->qtype),
                r->nrows,r->nin,r->ith,r->nth,r->tid,r->cpu,r->end_cpu,r->start,r->prep_in,r->prep_out,r->finish,r->released,r->sync);
            for (int k=0;k<r->nexperts;k++) {
                const struct wb_expert *x=&r->e[k];
                fprintf(f,"%s%d,%d,%d,%d,%d,%d,%"PRIu64",%"PRIu64,k?";":"",x->id,x->inputs,x->compact,x->rows,x->chunks,x->tiled,x->start,x->end);
            }
            fputc('\n',f);
        }
    }
    int bad=ferror(f);
    if (fclose(f)!=0 || bad) return -2;
    return atomic_load(&wb_graph);
}
