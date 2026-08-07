# wmforge 发布备忘（Release Checklist）

> 从 v0.1.0 发布流程沉淀。改版本号发新版时照此执行。
> 全流程预计 10 分钟，命令统一用 `/home/lanpeng/frappe-bench/env/bin/python`（该 venv 已有
> build/twine/pytest 及全部依赖）。

## 0. 前置

- [ ] 测试通过：`cd /home/lanpeng/wmforge && /home/lanpeng/frappe-bench/env/bin/python -m pytest tests/ -q`
  - 期望 25/25 PASS。**连跑 3 次确认稳定**（SVD 浮点存在跨依赖版本差异，曾有临界用例在
    numpy 2.5/opencv 5.0 下翻车）。
  - 注意：环境越新（numpy>=2.5 / opencv>=5.0），55%+JPEG 类临界测试余量越小。若失败，
    先重跑确认是否偶发，再考虑调测试参数（JPEG 质量 60→70 是 v0.1.0 的已知对策，留 18 byte 余量）。
- [ ] 准备 GitHub token（`repo` 权限足够）和 PyPI token（scope 选 wmforge 项目）。
  **token 不要直接贴到聊天记录里**——用后立即吊销，下次发布重新生成。

## 1. 改版本号（两处，必须同步）

| 文件 | 位置 |
|------|------|
| `pyproject.toml` | `[project] version = "0.1.0"` → 新版本 |
| `wmforge/__init__.py` | `__version__ = "0.1.0"` → 新版本 |

- [ ] 顺手核对 `pyproject.toml`：`license = "MIT"`（SPDX 表达式，**不要再加
      `License :: OSI Approved :: MIT License` classifier**——新版 setuptools 按 PEP 639
      报冲突，build 直接失败）。
- [ ] CHANGELOG / README 如有版本相关描述同步更新。

## 2. 构建验证（本地，不碰网络）

```bash
cd /home/lanpeng/wmforge
/home/lanpeng/frappe-bench/env/bin/python -m build
/home/lanpeng/frappe-bench/env/bin/python -m twine check dist/*
```

- [ ] build 无报错，`dist/` 出现 `wmforge-<ver>-py3-none-any.whl` + `wmforge-<ver>.tar.gz`
- [ ] twine check 输出两个 `PASSED`

## 3. Git 提交 + 打 tag

```bash
cd /home/lanpeng/wmforge
git add -A
git -c user.name="RevelryAge" -c user.email="RevelryAge@users.noreply.github.com" \
    commit -m "release: wmforge v<新版本>"
git -c user.name="RevelryAge" -c user.email="RevelryAge@users.noreply.github.com" \
    tag -a v<新版本> -m "wmforge v<新版本>"
```

- [ ] `git status --short` 确认没有 `_tmp_*`、`*.egg-info/`、`__pycache__/`、`dist/` 入库
      （`.gitignore` 已覆盖，提交前目检一次）
- [ ] 提交信息按约定写 `release: wmforge vX.Y.Z`

## 4. 推送 GitHub（必须走代理）

本机直连 github.com 的 TLS 被墙，**必须显式走代理 `http://192.168.2.33:8080`**；
认证格式用 **token 当用户名**（`oauth2:` / `x-access-token:` 前缀都试过不行）：

```bash
cd /home/lanpeng/wmforge
git -c http.proxy=http://192.168.2.33:8080 \
    push https://<GITHUB_TOKEN>@github.com/RevelryAge/wmforge.git main
git -c http.proxy=http://192.168.2.33:8080 \
    push https://<GITHUB_TOKEN>@github.com/RevelryAge/wmforge.git v<新版本>
```

- [ ] 两条 push 都成功（可用 `curl -sS https://api.github.com/repos/RevelryAge/wmforge/tags`
      验证 tag 已上线）
- [ ] 注意：`origin` remote 的 URL 保持干净（`https://github.com/RevelryAge/wmforge.git`），
      token 只用一次性 push URL 传递，不落盘、不进 `.git/config`

## 5. 发布 PyPI

```bash
cd /home/lanpeng/wmforge
/home/lanpeng/frappe-bench/env/bin/python -m twine upload -u __token__ -p "<PYPI_TOKEN>" dist/*
```

- [ ] 输出末尾出现 `View at: https://pypi.org/project/wmforge/<版本>/` 即成功
- [ ] **不要先试 TestPyPI**：TestPyPI 与 PyPI 是两套独立账号体系，pypi.org 的 token
      在 test.pypi.org 必 403。真要试发需在 test.pypi.org 单独注册 + 单独建 token。

## 6. 发布后验证 + 安全收尾

- [ ] 官方源验证（**用官方源，不用清华镜像**——镜像对新包有同步延迟，可能几小时）：
  ```bash
  /home/lanpeng/frappe-bench/env/bin/python -m pip install --no-deps -t /tmp/wmtest \
      --index-url https://pypi.org/simple/ wmforge==<新版本>
  PYTHONPATH=/tmp/wmtest /home/lanpeng/frappe-bench/env/bin/python -c "import wmforge; print(wmforge.__version__)"
  rm -rf /tmp/wmtest
  ```
- [ ] 检查 `https://pypi.org/pypi/wmforge/<版本>/json` 元数据（version/license/author/urls）
- [ ] **吊销用过的 GitHub token 和 PyPI token**（都可能在命令行/终端历史里暴露过），
      清 shell 历史（`history -c` 或重启终端）

## 已知坑速查

| 症状 | 原因 | 对策 |
|------|------|------|
| `git push` TLS 握手中断 | github.com 被墙 | 加 `-c http.proxy=http://192.168.2.33:8080` |
| push 认证 401 | 认证前缀用错 | 用 `https://<token>@github.com/...`（token 当用户名） |
| push 报缺密码（`could not read Password`） | URL 里写 `$VAR` 但 `VAR=x git push` 不会把变量传进 URL | token 直接硬编码进 URL，或先 `export` 再 push |
| build 报 PEP 639 license 冲突 | `license` 字段与 License classifier 并存 | 删掉 classifier，只留 `license = "MIT"` |
| TestPyPI 403 | token 是 pypi.org 的 | 跳过 TestPyPI，直接发 PyPI |
| pip 找不到新版本 | 清华镜像同步延迟 | 临时 `--index-url https://pypi.org/simple/` |
| 临界测试偶发失败 | SVD 浮点跨依赖版本差异 | 重跑确认；调测试 JPEG 质量留余量 |
| tag 创建报缺身份 | git 未配置 user.name/email | 用 `-c user.name=... -c user.email=...` 单次参数（勿改全局 config） |
