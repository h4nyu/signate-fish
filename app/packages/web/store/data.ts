import { observable, computed } from "mobx";
import { Map, List } from "immutable";
import { Workspaces } from ".";
import { ToastStore } from "./toast";
import { LoadingStore } from "./loading";
import { RootApi } from "@sivic/api";
import { Workspace } from "@sivic/core/workspace";
import { saveAs } from "file-saver";
import { MemoryRouter } from "react-router";
import { take, flow, sortBy, map } from "lodash/fp";
import { parseISO } from "date-fns";

export type State = {
  workspaces: Workspaces;
};

export type DataStore = {
  state: State;
  init: () => Promise<void>;
  fetchWorkspace: (id: string) => Promise<void>;
};

const State = (): State => {
  return {
    workspaces: List(),
  };
};

export const DataStore = (args: {
  api: RootApi;
  loading: <T>(fn: () => Promise<T>) => Promise<T>;
  toast: ToastStore;
}): DataStore => {
  const { api, loading, toast } = args;
  const state = observable(State());

  const fetchWorkspaces = async (): Promise<void> => {
    await loading(async () => {
      const rows = await api.workspace.filter({});
      if (rows instanceof Error) {
        return;
      }
      state.workspaces = List(rows);
    });
  };

  const fetchWorkspace = async (id: string): Promise<void> => {
    await loading(async () => {
      const row = await api.workspace.find({ id });
      if (row instanceof Error) {
        return;
      }
      state.workspaces = state.workspaces.filter((x) => x.id !== id);
      state.workspaces = state.workspaces.push(row);
    });
  };

  const init = async () => {
    await loading(async () => {
      await fetchWorkspaces();
    });
  };

  return {
    state,
    init,
    fetchWorkspace,
  };
};
