"""Download Titanic competition data using the refreshed OAuth token."""
import os
import sys

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk import KaggleClient, KaggleCredentials

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main() -> None:
    # Refresh OAuth access token from credentials.json
    client = KaggleClient()
    creds = KaggleCredentials.load(client=client)
    if creds is None:
        sys.exit("No OAuth credentials found in ~/.kaggle/credentials.json")
    token = creds.get_access_token()
    print(f"Authenticated as: {creds.get_username()}")

    # Build KaggleApi with the OAuth token
    api = KaggleApi()
    api._load_config()
    api.config_values[api.CONFIG_NAME_TOKEN] = token
    api.config_values[api.CONFIG_NAME_USER] = creds.get_username()
    api.config_values[api.CONFIG_NAME_AUTH_METHOD] = "oauth"
    api._authenticated = True

    print(f"Downloading Titanic competition files to {OUT_DIR} ...")
    api.competition_download_files("titanic", path=OUT_DIR, quiet=False)
    print("Download complete.")


if __name__ == "__main__":
    main()