import { v4 as uuid } from "uuid";
import { Lock, Store, ErrorKind } from "@sivic/core";
import { Image as CharImage } from "@charpoints/core/image";
import { Workspace } from "@sivic/core/workspace";
import { Box as CharBox } from "@charpoints/core/box";
import { Service as WorkspaceService } from "@sivic/core/workspace";

export const ImageTag = {
  Source: "Source",
  Target: "Target",
} as const;

export type ImageTag = typeof ImageTag[keyof typeof ImageTag];

export type Image = CharImage & {
  workspaceId: string;
  tag?: ImageTag;
};
export const Image = (): Image => {
  return {
    ...CharImage(),
    tag: undefined,
    workspaceId: "",
  };
};

export type CreatePayload = {
  name: string;
  workspaceId: string;
  data: string; //base64
};

export type UpdatePayload = {
  id: string;
  name: string;
  workspaceId: string;
  data: string; //base64
  tag: ImageTag;
};

export type DeletePayload = {
  id: string;
};

export type FindPayload = {
  id: string;
};

export type DetectBoxPayload = { imageId: string };

type DetectedBoxes = {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
  confidence: number;
}[];
export type DetectBoxes = (payload: {
  data: string;
}) => Promise<[DetectedBoxes, string] | Error>;

const DetectBoxFn = (args: {
  store: Store;
  detectBoxes: DetectBoxes;
  lock: Lock;
}) => {
  const { store, detectBoxes } = args;
  const services = {
    image: Service(args),
  };
  return async (payload: { imageId: string }) => {
    const image = await services.image.find({ id: payload.imageId });
    if (image instanceof Error) {
      return image;
    }
    const detectRes = await detectBoxes({ data: image.data || "" });
    if (detectRes instanceof Error) {
      return detectRes;
    }
    const [boxes, data] = detectRes;
    image.data = data;
    const updateErr = await store.image.update(image);
    if (updateErr instanceof Error) {
      return updateErr;
    }
    return boxes.map((x) => {
      return { ...x, imageId: image.id };
    });
  };
};

export type Service = {
  create: (payload: CreatePayload) => Promise<string | Error>;
  update: (payload: UpdatePayload) => Promise<string | Error>;
  delete: (payload: DeletePayload) => Promise<string | Error>;
  find: (payload: FindPayload) => Promise<Image | Error>;
};

export const Service = (args: { store: Store; lock: Lock }): Service => {
  const { store, lock } = args;
  const services = {
    workspace: WorkspaceService(args),
  };
  const find = async (payload: FindPayload) => {
    const image = await store.image.find(payload);
    if (image instanceof Error) {
      return image;
    }
    if (image === undefined) {
      return new Error(ErrorKind.ImageNotFound);
    }
    return image;
  };

  const create = async (payload: CreatePayload) => {
    return await lock.auto(async () => {
      const workspace = await services.workspace.find({
        id: payload.workspaceId,
      });
      if (workspace instanceof Error) {
        return workspace;
      }
      const row = Image();
      row.data = payload.data;
      row.name = payload.name;
      row.workspaceId = payload.workspaceId;
      const err = await store.image.insert(row);
      if (err instanceof Error) {
        return err;
      }
      return row.id;
    });
  };

  const update = async (payload: UpdatePayload) => {
    return await lock.auto(async () => {
      const workspace = await services.workspace.find({
        id: payload.workspaceId,
      });
      if (workspace instanceof Error) {
        return workspace;
      }
      const row = await find({ id: payload.id });
      if (row instanceof Error) {
        return row;
      }
      const newRow = {
        ...row,
        ...payload,
        updatedAt: new Date(),
      };
      const err = await store.image.update(newRow);
      if (err instanceof Error) {
        return err;
      }
      return row.id;
    });
  };

  const delete_ = async (payload: DeletePayload) => {
    return await lock.auto(async () => {
      const { id } = payload;
      const row = await find({ id });
      if (row instanceof Error) {
        return row;
      }
      const err = await store.image.delete({ id });
      if (err instanceof Error) {
        return err;
      }
      return row.id;
    });
  };

  return {
    create,
    update,
    find,
    delete: delete_,
  };
};
