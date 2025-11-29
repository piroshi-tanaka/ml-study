# 02. WSL Ubuntu環境構築手順

## 📋 目次

1. [概要](#概要)
2. [前提条件](#前提条件)
3. [WSLの確認とインストール](#wslの確認とインストール)
4. [Ubuntuのインストール](#ubuntuのインストール)
5. [Ubuntu初期設定](#ubuntu初期設定)
6. [ONNX Runtime C++のインストール](#onnx-runtime-cのインストール)
7. [VS Code + WSL拡張機能の設定](#vs-code--wsl拡張機能の設定)
8. [トラブルシューティング](#トラブルシューティング)

---

## 概要

### 目的

Windows PC上のWSL Ubuntu環境で、C++からONNX推論を実行できる環境を構築します。

### なぜWSL Ubuntu？

- **AUTOSAR環境の検証**: Linux環境でC++推論を検証
- **開発効率**: Windows上でLinux環境を利用
- **互換性**: 実際の組み込みLinux環境に近い

### 成果物

構築後、以下が使えるようになります：

- ✅ WSL Ubuntu 22.04 LTS
- ✅ ONNX Runtime C++ API (v1.18.0)
- ✅ ビルドツール（gcc, g++, cmake）
- ✅ VS Code + WSL拡張機能

### 所要時間

- 初回: 約30-40分
- WSL/Ubuntuインストール済みの場合: 約15-20分

---

## 前提条件

### システム要件

- **OS**: Windows 10 (バージョン2004以降) または Windows 11
- **CPU**: 仮想化対応（Intel VT-x / AMD-V）
- **メモリ**: 8GB以上推奨
- **ディスク容量**: 約10GB以上の空き容量

### 必要な権限

- 管理者権限（一部の操作で必要）

### 前提知識

- 基本的なコマンドライン操作
- 特別なLinux知識は不要（このドキュメントで説明）

---

## WSLの確認とインストール

### ステップ1: WSLの確認

**PowerShellを開く**:
1. `Windows キー` を押す
2. 「powershell」と入力
3. 「Windows PowerShell」をクリック

**確認コマンド実行**:
```powershell
wsl --version
```

**結果の判定**:

#### ✅ Case A: バージョン情報が表示される

```
WSL バージョン: 2.0.14.0
カーネル バージョン: 5.15.133.1-1
...
```

→ WSLインストール済み。[Ubuntuのインストール](#ubuntuのインストール)へ進む

#### ❌ Case B: エラーが表示される

```
'wsl' は、内部コマンドまたは外部コマンド、
操作可能なプログラムまたはバッチ ファイルとして認識されていません。
```

→ WSLが未インストール。次のステップへ

### ステップ2: WSLのインストール（未インストールの場合）

**PowerShell（管理者権限）を開く**:
1. `Windows キー` を押す
2. 「powershell」と入力
3. 「Windows PowerShell」を**右クリック**
4. 「管理者として実行」を選択
5. 「はい」をクリック

**インストールコマンド実行**:
```powershell
wsl --install
```

**期待される出力**:
```
インストール中: 仮想マシン プラットフォーム
インストール中: Linux 用 Windows サブシステム
...
要求された操作は正常に完了しました。変更を有効にするには、システムを再起動する必要があります。
```

**PCを再起動**:
```powershell
Restart-Computer
```

または手動で再起動

### ステップ3: WSL2の確認（推奨）

再起動後、PowerShellで：

```powershell
wsl --version
```

`WSL バージョン: 2.x.x.x` と表示されればOK

---

## Ubuntuのインストール

### ステップ1: インストール可能なディストリビューション確認

```powershell
wsl --list --online
```

**期待される出力**:
```
NAME                                   FRIENDLY NAME
Ubuntu                                 Ubuntu
Ubuntu-22.04                           Ubuntu 22.04 LTS
Ubuntu-20.04                           Ubuntu 20.04 LTS
...
```

### ステップ2: Ubuntu 22.04のインストール

**方法A: コマンドラインでインストール（推奨）**

```powershell
wsl --install -d Ubuntu-22.04
```

**期待される出力**:
```
Ubuntu 22.04 LTS をダウンロードしています...
インストール中...
```

**所要時間**: 5-10分

**方法B: Microsoft Storeからインストール**

1. `Windows キー` → 「Microsoft Store」を開く
2. 検索ボックスに「Ubuntu 22.04」と入力
3. 「Ubuntu 22.04.3 LTS」を選択
4. 「入手」または「インストール」をクリック
5. インストール完了後「起動」をクリック

### ステップ3: インストールの確認

```powershell
wsl --list --verbose
```

**期待される出力**:
```
  NAME            STATE           VERSION
* Ubuntu-22.04    Running         2
```

- `*` = デフォルトのディストリビューション
- `VERSION` が `2` = WSL2（推奨）

---

## Ubuntu初期設定

### ステップ1: Ubuntuの起動

**方法A: PowerShellから起動**

```powershell
wsl
```

**方法B: スタートメニューから起動**

1. `Windows キー` を押す
2. 「Ubuntu」と入力
3. 「Ubuntu 22.04 LTS」をクリック

### ステップ2: 初回起動時の設定

**初回起動時のみ、以下が表示されます**:

```
Installing, this may take a few minutes...
Please create a default UNIX user account. The username does not need to match your Windows username.
```

**ユーザー名を入力**:
```
Enter new UNIX username: horih
```

💡 **Tips**:
- 英小文字を推奨
- 簡潔なものが良い
- Windowsのユーザー名と同じである必要はない

**パスワードを入力**:
```
New password: ********
```

⚠️ **重要**:
- **画面には何も表示されません**（セキュリティのため）
- タイプして `Enter` を押してください
- 覚えやすいパスワードを設定

**パスワードを再入力**:
```
Retype new password: ********
```

**完了メッセージ**:
```
Installation successful!
To run a command as administrator (user "root"), use "sudo <command>".
See "man sudo_root" for details.

horih@DESKTOP-XXXXX:~$
```

✅ このプロンプトが表示されればOK！

### ステップ3: 基本的な動作確認

**現在地を確認**:
```bash
pwd
```

**期待される出力**:
```
/home/horih
```

これがあなたの「ホームディレクトリ」です。

**バージョン確認**:
```bash
lsb_release -a
```

**期待される出力**:
```
Distributor ID: Ubuntu
Description:    Ubuntu 22.04.3 LTS
Release:        22.04
Codename:       jammy
```

✅ Ubuntu 22.04と表示されればOK

### ステップ4: システムのアップデート

```bash
# パッケージリストの更新
sudo apt update
```

**初回実行時**: パスワード入力を求められます（画面には表示されません）

**期待される出力**:
```
Hit:1 http://archive.ubuntu.com/ubuntu jammy InRelease
...
All packages are up to date.
```

```bash
# インストール済みパッケージのアップグレード
sudo apt upgrade -y
```

**期待される出力**:
```
Reading package lists... Done
Building dependency tree... Done
...
0 upgraded, 0 newly installed, 0 to remove and 0 not upgraded.
```

**所要時間**: 5-10分

✅ エラーなく完了すればOK

---

## ONNX Runtime C++のインストール

### ステップ1: 必要なツールのインストール

```bash
# ビルドツールのインストール
sudo apt install -y build-essential cmake wget
```

**期待される出力**:
```
Reading package lists... Done
...
Setting up build-essential...
Setting up cmake...
Setting up wget...
```

**インストールされるツール**:
- `gcc/g++`: C++コンパイラ
- `cmake`: ビルドシステム
- `wget`: ファイルダウンローダー

### ステップ2: ONNX Runtime のダウンロード

```bash
# ホームディレクトリに移動
cd ~

# ONNX Runtime v1.18.0 をダウンロード
wget https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz
```

**期待される出力**:
```
--2025-11-02 16:00:00--  https://github.com/microsoft/...
Resolving github.com... 20.27.177.113
Connecting to github.com|20.27.177.113|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 5740642 (5.5M) [application/x-gzip]
Saving to: 'onnxruntime-linux-x64-1.18.0.tgz'

onnxruntime-linux-x64-1.18.0.tgz  100%[======>] 5.47M  --.-KB/s    in 0.08s

2025-11-02 16:00:01 (68.9 MB/s) - 'onnxruntime-linux-x64-1.18.0.tgz' saved
```

**所要時間**: 数秒〜1分

### ステップ3: 解凍

```bash
tar -xzf onnxruntime-linux-x64-1.18.0.tgz
```

**確認**:
```bash
ls onnxruntime-linux-x64-1.18.0/
```

**期待される出力**:
```
GIT_COMMIT_ID  LICENSE  Privacy.md  README.md  ThirdPartyNotices.txt  VERSION_NUMBER  include  lib
```

✅ `include` と `lib` ディレクトリがあればOK

### ステップ4: システムディレクトリにインストール

#### 4-1. ヘッダーファイルのインストール

```bash
# onnxruntimeディレクトリを作成
sudo mkdir -p /usr/local/include/onnxruntime

# ヘッダーファイルをコピー
sudo cp -v ~/onnxruntime-linux-x64-1.18.0/include/*.h /usr/local/include/onnxruntime/
```

**期待される出力**:
```
'/home/horih/onnxruntime-linux-x64-1.18.0/include/cpu_provider_factory.h' -> '/usr/local/include/onnxruntime/cpu_provider_factory.h'
'/home/horih/onnxruntime-linux-x64-1.18.0/include/onnxruntime_c_api.h' -> '/usr/local/include/onnxruntime/onnxruntime_c_api.h'
'/home/horih/onnxruntime-linux-x64-1.18.0/include/onnxruntime_cxx_api.h' -> '/usr/local/include/onnxruntime/onnxruntime_cxx_api.h'
...（12個のファイル）
```

#### 4-2. ライブラリファイルのインストール

```bash
# ライブラリファイルをコピー
sudo cp -v ~/onnxruntime-linux-x64-1.18.0/lib/* /usr/local/lib/
```

**期待される出力**:
```
'/home/horih/onnxruntime-linux-x64-1.18.0/lib/libonnxruntime.so' -> '/usr/local/lib/libonnxruntime.so'
'/home/horih/onnxruntime-linux-x64-1.18.0/lib/libonnxruntime.so.1.18.0' -> '/usr/local/lib/libonnxruntime.so.1.18.0'
```

#### 4-3. ライブラリキャッシュの更新

```bash
sudo ldconfig
```

### ステップ5: インストールの確認

**ヘッダーファイルの確認**:
```bash
ls /usr/local/include/onnxruntime/
```

**期待される出力**:
```
cpu_provider_factory.h
onnxruntime_c_api.h
onnxruntime_cxx_api.h
onnxruntime_cxx_inline.h
...
```

**ライブラリファイルの確認**:
```bash
ls -l /usr/local/lib/libonnxruntime*
```

**期待される出力**:
```
lrwxrwxrwx 1 root root       25 Nov  2 16:30 /usr/local/lib/libonnxruntime.so -> libonnxruntime.so.1.18.0
-rwxr-xr-x 1 root root 18742568 Nov  2 16:30 /usr/local/lib/libonnxruntime.so.1.18.0
```

✅ 両方のファイルが存在すればOK！

---

## VS Code + WSL拡張機能の設定

### ステップ1: WSL拡張機能のインストール

**VS Codeを開く**（Windows側）

**拡張機能マーケットプレイスを開く**:
- サイドバーの拡張機能アイコン（四角が4つ）をクリック
- または `Ctrl + Shift + X`

**「WSL」で検索**

**インストール**:
- 名前: **WSL**
- 発行元: **Microsoft**
- 緑の「インストール」ボタンをクリック

### ステップ2: WSL環境でプロジェクトを開く

**Ubuntuターミナルで**:

```bash
# プロジェクトディレクトリに移動
cd /mnt/c/workspace/src/ml-study/scikit_learn_onnx_repo/cpp_inference

# VS Codeで開く
code .
```

**初回実行時のみ**: VS Code Serverのインストールが始まります（自動・1-2分）

**VS Codeが開いたら確認**:
- ✅ 左下に「**WSL: Ubuntu**」と表示されている
- ✅ エクスプローラーにファイル一覧が表示されている
- ✅ ターミナルがUbuntuになっている

### ステップ3: 統合ターミナルの確認

**ターミナルを開く**:
```
Ctrl + `
```

**プロンプトを確認**:
```
horih@DESKTOP-XXXXX:/mnt/c/.../cpp_inference$
```

✅ Ubuntu のプロンプトが表示されればOK

---

## トラブルシューティング

### Q1: WSLのインストールに失敗する

**エラー例**:
```
エラー: 0x80370102
仮想マシン プラットフォームの Windows 機能を有効にして、BIOS で仮想化が有効になっていることを確認してください。
```

**解決策**:

1. **Windows機能の有効化**:
   - `Windows キー` → 「Windowsの機能の有効化または無効化」
   - 以下を有効化:
     - ✅ Windows Subsystem for Linux
     - ✅ 仮想マシン プラットフォーム
   - PCを再起動

2. **BIOSで仮想化を有効化**:
   - PCを再起動してBIOSに入る
   - Intel VT-x または AMD-V を有効化
   - 保存して再起動

### Q2: Ubuntuの起動に失敗する

**エラー例**:
```
WslRegisterDistribution failed with error: 0x8007019e
```

**解決策**:
```powershell
# WSLカーネルの更新
wsl --update

# PCを再起動
Restart-Computer
```

### Q3: パスワード入力で何も表示されない

**現象**: パスワード入力時に文字が表示されない

**これは正常です！**
- セキュリティのため、画面には何も表示されません
- タイプして `Enter` を押してください

### Q4: `sudo apt update` でエラー

**エラー例**:
```
E: Could not get lock /var/lib/apt/lists/lock
```

**解決策**:
```bash
# 数秒待ってから再実行
sudo apt update

# それでもダメな場合
sudo rm /var/lib/apt/lists/lock
sudo rm /var/cache/apt/archives/lock
sudo apt update
```

### Q5: ダウンロードが遅い

**解決策**:
```bash
# ミラーサーバーを日本に変更（オプション）
sudo sed -i.bak -e 's|http://archive.ubuntu.com|http://jp.archive.ubuntu.com|g' /etc/apt/sources.list
sudo apt update
```

### Q6: `/usr/local/include` にコピーできない

**エラー例**:
```
Permission denied
```

**解決策**:
```bash
# sudoを忘れずに
sudo cp -v ~/onnxruntime-linux-x64-1.18.0/include/*.h /usr/local/include/onnxruntime/
```

### Q7: VS Codeで「code .」が使えない

**エラー例**:
```
code: command not found
```

**解決策**:
```bash
# VS Codeがインストールされているか確認（Windows側）
# 一度VS Codeを再起動
# Ubuntuターミナルから再実行
code .
```

---

## ✅ チェックリスト

環境構築が完了したら、以下を確認してください：

### WSL/Ubuntu

- [ ] `wsl --version` でWSL2のバージョンが表示される
- [ ] `wsl --list` でUbuntu-22.04が表示される
- [ ] Ubuntuが起動できる（`wsl`コマンドまたはスタートメニュー）
- [ ] ユーザー名とパスワードが設定されている
- [ ] `lsb_release -a` でUbuntu 22.04と表示される

### システムアップデート

- [ ] `sudo apt update` がエラーなく完了する
- [ ] `sudo apt upgrade -y` がエラーなく完了する

### ビルドツール

- [ ] `gcc --version` でバージョンが表示される
- [ ] `g++ --version` でバージョンが表示される
- [ ] `cmake --version` でバージョンが表示される

### ONNX Runtime

- [ ] `/usr/local/include/onnxruntime/` ディレクトリが存在する
- [ ] `/usr/local/include/onnxruntime/onnxruntime_cxx_api.h` ファイルが存在する
- [ ] `/usr/local/lib/libonnxruntime.so` ファイルが存在する
- [ ] `ldconfig -p | grep onnxruntime` でライブラリが表示される

### VS Code

- [ ] WSL拡張機能がインストールされている
- [ ] `code .` でVS Codeが開く
- [ ] 左下に「WSL: Ubuntu」と表示される
- [ ] 統合ターミナルでUbuntuが使える

---

## 📚 次のステップ

✅ **このステップが完了したら、次のドキュメントへ進んでください：**

👉 [**03_C++推論実行手順.md**](./03_C++推論実行手順.md)

C++でONNX推論を実行し、Python推論との精度を比較します。

---

## 📊 補足情報

### Linuxディレクトリ構造の基本

```
/ (ルート)
├── home/
│   └── horih/          ← あなたのホームディレクトリ
├── usr/
│   └── local/
│       ├── include/    ← ヘッダーファイル（システム共通）
│       └── lib/        ← ライブラリファイル（システム共通）
├── mnt/
│   └── c/              ← WindowsのCドライブ
└── tmp/                ← 一時ファイル
```

### なぜ `/usr/local/` に置くのか？

1. **システム全体から利用可能**
   - どのプロジェクトからでも使える
   - コンパイラが自動的に見つけられる

2. **標準的な場所**
   - ユーザーがインストールしたソフトウェアの置き場所
   - パッケージマネージャーと競合しない

3. **CMakeが自動認識**
   - デフォルトの検索パスに含まれている

### WSL2のメリット

1. **高速なファイルI/O**
   - WSL1より大幅に高速

2. **完全なLinux互換性**
   - 実際のLinuxカーネルを使用

3. **Dockerサポート**
   - Docker Desktopとの統合

### よく使うコマンド

```bash
# 現在地を確認
pwd

# ファイル一覧を表示
ls
ls -la  # 詳細表示

# ディレクトリ移動
cd /path/to/directory
cd ..  # 一つ上に戻る
cd ~   # ホームディレクトリに戻る

# ファイルの中身を表示
cat filename

# コマンド実行を中断
Ctrl + C

# 画面をクリア
clear

# Ubuntuを終了
exit
```

---

**作成日**: 2025-11-02  
**バージョン**: 1.0  
**対象**: WSL/Ubuntu初心者

