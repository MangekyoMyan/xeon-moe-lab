# Third-party materials

For a readable account of the projects, contributors, and research that informed this work, see [Acknowledgments / 謝辞・参考にしたOSSと研究](ACKNOWLEDGMENTS.md). It distinguishes code used in the implementation from design references and experiments; it does not replace the original license notices.

- The upstream source snapshot is from https://github.com/ggml-org/llama.cpp at commit `dde879bc4f954b809892fd0d660515f42536394a`. Its [LICENSE](vendor/llama.cpp/LICENSE) and bundled third-party license notices remain in their original locations.
- Local modified llama.cpp/ggml files retain their source notices and are mapped in `accepted/ORIGINS.json`. Historical research directories can contain copies and patches derived from the same upstream.
- CUDA toolkit binaries, model weights, and papers are not redistributed in this source archive. Runtime release assets contain this project's compiled binaries; external CUDA and OS libraries must be obtained separately under their own licenses.
- This repository preserves experimental source and evidence for a specific hardware configuration. A license for the project's original additions has not yet been selected; publication does not change third-party licenses or assert a new blanket license for unrelated material or benchmark inputs.
