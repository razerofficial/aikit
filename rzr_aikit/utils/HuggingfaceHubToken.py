from huggingface_hub import get_token, login, logout

class HuggingfaceHubToken:
    def __init__(self) :
        self.access_token = None

    def get_access_token(self) -> str:
        if self.access_token:
            return self.access_token

        # already login
        self.access_token = get_token()
        if self.access_token:
            return self.access_token

        try:
            from requests.exceptions import HTTPError

            # login
            login()

            # get token
            self.access_token = get_token()
            if self.access_token:
                return self.access_token

        except HTTPError as e:
            if (e.response.status_code == 401) and ("unauthorized" in e.response.reason.lower()):
                from rzr_aikit import HuggingfaceAccessTokenRequired
                raise HuggingfaceAccessTokenRequired("Huggingface Access Token is required") from e
            else:
                raise

    def delete_access_token(self):
        logout()
