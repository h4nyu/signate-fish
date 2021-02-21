import { observable, computed } from "mobx";
import { Map, List } from "immutable";
import { Workspaces, Images } from "@sivic/web/store";
import { ToastStore } from "./toast";
import { LoadingStore } from "./loading";
import { RootApi } from "@sivic/api";
import { Workspace } from "@sivic/core/workspace";
import { saveAs } from "file-saver";
import { MemoryRouter } from "react-router";
import { take, flow, sortBy, map } from "lodash/fp";
import { parseISO } from "date-fns";
import { Level } from "@sivic/web/store";
import { readAsBase64, b64toBlob } from "@charpoints/web/utils";
import { Image, ImageTag } from "@sivic/core/image";

export type State = {
  workspace?: Workspace;
  images: Images;
};

export type ImageForm = {
  state: State;
  init: (workspace: Workspace) => Promise<void>;
  updateTag: (imageId: string, tag:ImageTag) => void;
  uploadFiles: (files: File[]) => Promise<void>;
  deleteImage: (imageId: string) => Promise<void | Error>;
};

const State = (): State => {
  return {
    images: Map(),
  };
};

export const ImageForm = (args: {
  api: RootApi;
  loading: <T>(fn: () => Promise<T>) => Promise<T>;
  toast: ToastStore;
  onSave?: (workspaceId: string) => void;
}): ImageForm => {
  const { api, loading, toast, onSave } = args;
  const state = observable(State());
  const fetch = async () => {
    const { workspace } = state;
    if (workspace === undefined) {
      return;
    }
    const { imageIds } = workspace;
    const images: Image[] = [];
    for (const id of imageIds) {
      const image = await api.image.find({ id });
      if (image instanceof Error) {
        continue;
      }
      images.push(image);
    }
    state.images = images;
  };

  const init = async (workspace: Workspace) => {
    state.workspace = workspace;
    await fetch();
  };

  const updateTag = async (imageId: string, tag:ImageTag) => {
  };

  const deleteImage = async (imageId: string) => {
    const workspaceId = state.workspace?.id;
    if (workspaceId === undefined) {
      return;
    }
    const err = await api.image.delete({
      id: imageId,
    });
    if (err instanceof Error) {
      toast.error(err);
      return err;
    }
    onSave && onSave(workspaceId);
  };

  const uploadFiles = async (files: File[]) => {
    const ids: string[] = [];
    const workspaceId = state.workspace?.id;
    if (workspaceId === undefined) {
      return;
    }
    await loading(async () => {
      for (const f of files) {
        if (!f.type.includes("image")) {
          toast.error(Error("UnsupportedFormat"));
          continue;
        }
        const data = await readAsBase64(f);
        if (data instanceof Error) {
          toast.error(data);
          continue;
        }
        const imageId = await api.image.create({
          data,
          name: f.name,
          workspaceId,
        });
        if (imageId instanceof Error) {
          toast.error(imageId);
          continue;
        }
      }
    });
    onSave && onSave(workspaceId);
  };

  return {
    state,
    init,
    deleteImage,
    uploadFiles,
    updateTag,
  };
};
export default ImageForm;
