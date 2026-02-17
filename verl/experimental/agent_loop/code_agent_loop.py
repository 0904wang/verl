# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Code-agent flavored loop.

This loop currently reuses ToolAgentLoop behavior and only registers a new
agent name so users can switch with config.
"""

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop


@register("code_agent")
class CodeAgentLoop(ToolAgentLoop):
    """Code agent loop alias based on ToolAgentLoop."""

