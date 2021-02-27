import { Sql } from "postgres";
import { Store, Lock } from "@sivic/core";
import { FastifyPlugin } from "fastify";
import {
  Service,
  CreatePayload,
  FindPayload,
  DeletePayload,
  UpdatePayload,
} from "@sivic/core/image";

export const DetectRoutes = (args: {
  store: Store;
  lock: Lock;
}): FastifyPlugin<{ prefix: string }> => {
  const { store, lock } = args;
  const srv = Service({ store, lock });
  return function (app, opts, done) {
    app.post<{ Body: CreatePayload }>("/detect/box", {}, async (req, reply) => {
      const res = await srv.create(req.body);
      reply.send(res);
    });
    done();
  };
};
