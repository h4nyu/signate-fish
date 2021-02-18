import React, { Suspense, lazy } from "react";
import { Router, Switch, Route } from "react-router-dom";
import rootStore from "./store";
import Loading from "./connectors/Loading";
import Toast from "./connectors/Toast";
import { createHashHistory } from "history";
import PageLayout from "@sivic/web/components/PageLayout";
import Header from "@sivic/web/components/Header";
import Sidebar from "@sivic/web/components/Sidebar";

const history = createHashHistory();
const MainPage = lazy(() => import("@sivic/web/pages/MainPage"));
const WorkspacePage = lazy(() => import("@sivic/web/pages/WorkspacePage"));
const WorkspaceCreatePage = lazy(
  () => import("@sivic/web/pages/WorkspaceCreatePage")
);
const ImagePage = lazy(() => import("@sivic/web/pages/ImagePage"));

export default function App() {
  React.useEffect(() => {
    rootStore.init();
    history.push("/workspace");
  });
  return (
    <>
      <Router history={history}>
        <Loading />
        <Toast />
        <PageLayout
          header={<Header />}
          sidebar={<Sidebar />}
          content={
            <Suspense fallback={<div>Loading...</div>}>
              <Route exact path={"/workspace"} component={MainPage} />
              <Route
                exact
                path={"/workspace/id/:id"}
                component={WorkspacePage}
              />
              <Route
                exact
                path={"/workspace/create"}
                component={WorkspaceCreatePage}
              />
              <Route
                exact
                path={"/workspace/id/:id/image-id/:imageId"}
                component={ImagePage}
              />
            </Suspense>
          }
        />
      </Router>
    </>
  );
}
