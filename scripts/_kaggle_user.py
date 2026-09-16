"""현재 인증된 Kaggle username을 출력한다.

새 방식(~/.kaggle/access_token, KAGGLE_API_TOKEN)과 구 방식(kaggle.json) 모두 지원.
"""

import sys


def main() -> int:
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("[!] kaggle 패키지 없음: pip install -r requirements.txt", file=sys.stderr)
        return 1

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as e:  # noqa: BLE001
        print(
            f"[!] Kaggle 인증 실패: {e}\n"
            "    https://www.kaggle.com/settings/api 에서 토큰 발급 후:\n"
            "    mkdir -p ~/.kaggle && echo <KGAT_토큰> > ~/.kaggle/access_token "
            "&& chmod 600 ~/.kaggle/access_token",
            file=sys.stderr,
        )
        return 1

    username = api.config_values.get("username")
    if not username:
        print("[!] username을 확인할 수 없다. 토큰이 만료됐을 수 있다.", file=sys.stderr)
        return 1

    print(username)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
