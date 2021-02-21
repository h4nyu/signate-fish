import { DataStore } from "./data";
import { LoadingStore } from "./loading";
import { ToastStore } from "./toast";
import { RootApi } from "@sivic/api";
import { configure } from "mobx";
import { Map, List } from "immutable";
import { createHashHistory } from "history";
import { Workspace } from "@sivic/core/workspace";
import { Image } from "@sivic/core/image";
import WorkspaceForm from "@sivic/web/store/WorkspaceForm";
import ImageForm from "@sivic/web/store/ImageForm";
import ImageProcess from "@sivic/web/store/ImageProcess";

configure({
  enforceActions: "never",
});

export type Workspaces = List<Workspace>;
export type Images = Map<string, Image>;
export enum Level {
  Info,
  Success,
  Warning,
  Error,
}
export type LoadingFn = <T>(fn: () => Promise<T>) => Promise<T>;

export type History = {
  push: (name: string) => void;
  goBack: () => void;
};

export type RootStore = {
  data: DataStore;
  loading: LoadingStore;
  toast: ToastStore;
  history: History;
  api: RootApi;
  workspaceForm: WorkspaceForm;
  imageForm: ImageForm;
  imageProcess: ImageProcess;
  init: () => Promise<void>;
};
export const RootStore = (): RootStore => {
  const api = RootApi();
  const loading = LoadingStore();
  const toast = ToastStore();
  const data = DataStore({ api, loading: loading.loading, toast });
  const history = createHashHistory();

  const init = async () => {
    await data.init();
    toast.show("Success", Level.Success);
  };

  const imageForm = ImageForm({
    api,
    loading: loading.loading,
    toast,
    onSave: async (workspaceId: string) => {
      // workspaceForm.update(workspaceId);
    },
  });
  const imageProcess = ImageProcess({
    api,
    loading: loading.loading,
    toast,
    onInit: (workspaceId, imageId) => {
      history.push(`/workspace/id/${workspaceId}/image-id/${imageId}`);
    },
  });
  const workspaceForm = WorkspaceForm({
    api,
    loading: loading.loading,
    toast,
    imageForm,
    onUpdate: (workspace) => {
      history.push(`/workspace/id/${workspace.id}`);
    },
    onSave: (workspace) => {
      data.fetchWorkspace(workspace.id);
    },
    onDelete: (id: string) => {
      data.init();
    },
  });

  return {
    api,
    data,
    toast,
    loading,
    init,
    history,
    workspaceForm,
    imageForm,
    imageProcess,
  };
};

export default RootStore();
