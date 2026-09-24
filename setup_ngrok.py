import ssl
from pyngrok import conf, ngrok, installer

# SSL証明書検証を無効化（ダウンロードエラー回避）
ssl._create_default_https_context = ssl._create_unverified_context

# 先ほど管理画面でコピーした Authtoken をここに入力
AUTHTOKEN = "3Ig4GqeSrNzexyhSEJlDZj1Lc3H_7SBBx5ZBGnGyTSuvxf1Bc"

print("ngrok 本体をダウンロード中...")
pyngrok_config = conf.get_default()
installer.install_ngrok(pyngrok_config.ngrok_path)

print("Authtoken を設定中...")
ngrok.set_auth_token(AUTHTOKEN)

print("\n" + "=" * 50)
print("🚀 セットアップ完了！トンネルを起動します...")
public_url = ngrok.connect(8501)
print(f"📱 スマホから以下のURLにアクセスしてください:")
print(f"👉 {public_url}")
print("=" * 50 + "\n")

input("Press Enter to stop the tunnel...\n")