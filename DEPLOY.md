# RetainPro Web 部署说明

## 本地运行

```bash
cd /Users/dingxiangyue/Desktop/数字银行用户流失的分析与预测/run
python -m pip install -r requirements.txt
streamlit run app.py
```

也可以在项目根目录直接运行：

```bash
./start_web.command
```

关闭本地服务：

```bash
./stop_web.command
```

## 公网部署

推荐先用 Streamlit Community Cloud：

1. 把 `run/` 目录里的文件提交到 GitHub 仓库。
2. 在 Streamlit Community Cloud 新建应用。
3. Main file path 填 `app.py`。
4. 部署完成后会得到公网访问地址。

## 后续小程序路线

小程序前端不能直接运行 Python，需要把当前分析流程改成 FastAPI 接口，再让小程序通过 HTTPS 调用接口。当前 `app.py` 适合先做网页演示和业务验证。
