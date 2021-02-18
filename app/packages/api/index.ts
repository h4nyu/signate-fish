import { WorkspaceApi } from "./workspace";
import { ImageApi } from "./image";
import axios from "axios";

export function toError(err: any): Error {
  const message = err.response?.data?.message;
  if (message) {
    return new Error(message);
  } else {
    return new Error(err.message);
  }
}

export type RootApi = {
  setUrl: (url: string) => void;
  getImageStoreUrl: () => Promise<string | Error>;
  workspace: WorkspaceApi;
  image: ImageApi;
};

export const RootApi = (): RootApi => {
  const http = axios.create();
  const prefix = "api/v1";
  const workspace = WorkspaceApi({ http, prefix: `${prefix}/workspace` });
  const image = ImageApi({ http, prefix: `${prefix}/image` });

  const setUrl = (url: string) => {
    http.defaults.baseURL = url;
  };

  const getImageStoreUrl = async () => {
    try {
      const res = await http.get(`${prefix}/image-store-url`);
      return res.data;
    } catch (err) {
      return toError(err);
    }
  };
  return {
    setUrl,
    getImageStoreUrl,
    workspace,
    image,
  };
};
