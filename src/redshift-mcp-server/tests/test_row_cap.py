# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
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

"""Tests for the MCP_REDSHIFT_MAX_ROWS row cap on execute_query / execute_query_rw."""

import pytest
from awslabs.redshift_mcp_server.server import execute_query_rw_tool, execute_query_tool
from mcp.server.fastmcp import Context


def _data(n: int) -> dict:
    return {
        'columns': ['id'],
        'rows': [[i] for i in range(n)],
        'row_count': n,
        'query_id': 'q-1',
    }


async def _run(tool, mocker):
    mocker.patch('awslabs.redshift_mcp_server.server.execute_query', return_value=_data(120))
    return await tool(Context(), cluster_identifier='c', database_name='dev', sql='SELECT 1')


@pytest.mark.asyncio
@pytest.mark.parametrize('tool', [execute_query_tool, execute_query_rw_tool])
async def test_cap_truncates_rows_keeps_total(tool, mocker, monkeypatch):
    """Cap trims rows, flags truncated, keeps the true total in row_count."""
    monkeypatch.setenv('MCP_REDSHIFT_MAX_ROWS', '50')
    result = await _run(tool, mocker)
    assert len(result.rows) == 50
    assert result.truncated is True
    assert result.row_count == 120


@pytest.mark.asyncio
@pytest.mark.parametrize('env', [None, '0', ''])
async def test_cap_off_returns_all_rows(env, mocker, monkeypatch):
    """Unset, 0 or empty env var leaves results untouched."""
    if env is None:
        monkeypatch.delenv('MCP_REDSHIFT_MAX_ROWS', raising=False)
    else:
        monkeypatch.setenv('MCP_REDSHIFT_MAX_ROWS', env)
    result = await _run(execute_query_tool, mocker)
    assert len(result.rows) == 120
    assert result.truncated is False


@pytest.mark.asyncio
async def test_under_cap_not_truncated(mocker, monkeypatch):
    """Results under the cap are not flagged."""
    monkeypatch.setenv('MCP_REDSHIFT_MAX_ROWS', '500')
    result = await _run(execute_query_tool, mocker)
    assert len(result.rows) == 120
    assert result.truncated is False
