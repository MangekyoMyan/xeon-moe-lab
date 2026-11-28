# Third-party materials

For a readable account of the projects, contributors, and research that informed this work, see [Acknowledgments / 謝辞・参考にしたOSSと研究](ACKNOWLEDGMENTS.md). It distinguishes code used in the implementation from design references and experiments; it does not replace the original license notices.

- The upstream source snapshot is from https://github.com/ggml-org/llama.cpp at commit `dde879bc4f954b809892fd0d660515f42536394a`. Its [LICENSE](vendor/llama.cpp/LICENSE) and bundled third-party license notices remain in their original locations.
- Local modifications to llama.cpp and ggml retain the upstream notices. The file maps identify the sources used for each profile. [Current Xeon/Volta build](current/ORIGINS.json) · [November 5 baseline](accepted/ORIGINS.json).
- CUDA toolkit binaries, model weights, and papers are not redistributed in this source archive. Runtime release assets contain this project's compiled binaries; external CUDA and OS libraries must be obtained separately under their own licenses.
- The project's original code and documentation are provided under the [MIT License](LICENSE), copyright (c) 2026 MangekyouMyan. Third-party code, materials, and benchmark inputs retain their existing licenses and notices; the project license does not relicense those materials.
