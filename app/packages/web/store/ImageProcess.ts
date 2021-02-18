import { observable, computed } from "mobx";
import { Map, List } from "immutable";
import { Workspaces } from ".";
import { ToastStore } from "./toast";
import { LoadingStore } from "./loading";
import { RootApi } from "@sivic/api";
import { RootApi as ImageApi } from "@charpoints/api";
import { Workspace } from "@sivic/core/workspace";
import { Image } from "@charpoints/core/image";
import { saveAs } from "file-saver";
import { MemoryRouter } from "react-router";
import { take, flow, sortBy, map } from "lodash/fp";
import { parseISO } from "date-fns";
import { Level } from "@sivic/web/store";
import { readAsBase64, b64toBlob } from "@charpoints/web/utils";
import { Images } from "@charpoints/web/store";

export type State = {
  image?: Image;
};

export type ImageProcess = {
  state: State;
  init: (workspaceId: string, imageId: string) => Promise<void | Error>;
};

const State = (): State => {
  return {};
};

export const ImageProcess = (args: {
  api: RootApi;
  loading: <T>(fn: () => Promise<T>) => Promise<T>;
  toast: ToastStore;
  onInit?: (workspaceId: string, imageId: string) => void;
}): ImageProcess => {
  const { api, loading, toast, onInit } = args;
  const state = observable(State());

  const init = async (workspaceId: string, imageId: string) => {
    await loading(async () => {
      const image = await api.image.find({ id: imageId });
      if (image instanceof Error) {
        return image;
      }
      state.image = image;
      onInit && onInit(workspaceId, imageId);
    });
  };

  return {
    state,
    init,
  };
};
export default ImageProcess;
