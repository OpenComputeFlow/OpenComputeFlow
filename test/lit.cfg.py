import os

import lit.formats
from lit.llvm import llvm_config


config.name = "OpenComputeFlow"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir", ".test"]
config.excludes = ["CMakeLists.txt", "lit.cfg.py", "lit.site.cfg.py.in"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.ocf_obj_root, "test")

llvm_config.use_default_substitutions()
llvm_config.add_tool_substitutions(
    ["ocf-opt"], [config.ocf_tools_dir, config.llvm_tools_dir]
)
