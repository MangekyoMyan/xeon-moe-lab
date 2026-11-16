// Stable expert/token/slot order, matching the former expert-major scan exactly.
// Duplicate experts within one token retain the original completeness rejection.
static void xeon_sort_moe_ids(const char * ids, size_t nb0, size_t nb1,
        int64_t experts, int64_t tokens, int64_t used, int64_t input_columns,
        std::vector<int32_t> & to_sorted, std::vector<int32_t> & from_sorted,
        std::vector<int32_t> & tokens_per_expert) {
    std::vector<int64_t> offsets(experts, -1);
    for (int64_t t=0;t<tokens;++t) for (int64_t slot=0;slot<used;++slot) {
        int32_t expert;
        memcpy(&expert,ids+t*nb1+slot*nb0,sizeof(expert));
        assert(expert>=0 && expert<experts);
        if (expert>=0 && expert<experts && offsets[expert]!=t) {
            offsets[expert]=t;
            ++tokens_per_expert[expert];
        }
    }
    int64_t count=0;
    for (int64_t expert=0;expert<experts;++expert) {
        offsets[expert]=count;
        count+=tokens_per_expert[expert];
    }
    to_sorted.resize(count);
    GGML_ASSERT(to_sorted.size()==size_t(tokens*used));
    for (int64_t t=0;t<tokens;++t) for (int64_t slot=0;slot<used;++slot) {
        int32_t expert;
        memcpy(&expert,ids+t*nb1+slot*nb0,sizeof(expert));
        const int64_t position=offsets[expert]++;
        from_sorted[t*used+slot]=position;
        to_sorted[position]=t*input_columns+slot%input_columns;
    }
}
