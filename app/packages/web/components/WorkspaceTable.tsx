import React, { useState } from "react";
import { List } from "immutable";

import { Workspace } from "@sivic/core/workspace";
import DateView from "@sivic/web/components/DateView";
import TableHeader from "@sivic/web/components/TableHeader";
import DeleteBtn from "@sivic/web/components/DeleteBtn";
import CreateBtn from "@sivic/web/components/CreateBtn"

const columns = ["Name", "Create", "Action"];

const filterColumns = ["Name"];

export const WorkspaceTable = (props: {
  workspaces: Workspace[];
  onClick?: (id: string) => void;
  onCreate?:() => void; 
  onDelete?: (id: string) => void;
}) => {
  const { workspaces, onClick, onDelete, onCreate } = props;
  const [sort, setSort] = React.useState<[string, boolean]>(["Name", true]);
  const [sortColumn, asc] = sort;
  const [keyword, setKeyword] = useState("");
  const lowerKeyowerd = keyword.toLowerCase();

  let rows = List(workspaces)
    .map((x) => {
      return {
        ...x,
        Name: x.name,
        Create: x.createdAt,
        onClick: () => onClick && onClick(x.id),
        onDelete: () => onDelete && onDelete(x.id),
      };
    })
    .filter((x) =>
      filterColumns
        .map((c) => x[c])
        .join(" ")
        .toLowerCase()
        .includes(lowerKeyowerd)
    )
    .sortBy((x) => x[sortColumn]);
  if (asc) {
    rows = rows.reverse();
  }

  return (
      <div
        style={{
          display: "grid",
          gridTemplateRows: "auto 1fr",
          gridTemplateColumns: "1fr auto",
          width: "100%",
          height: "100%",
        }}
      >
        <input
          className="input"
          type="text"
          onChange={(e) => setKeyword(e.target.value)}
          style={{
            gridRow: "1",
            gridColumn: "1",
          }}
        />
        {
          onCreate && <CreateBtn 
            style={{
              gridRow: "1",
              gridColumn: "2",
            }}
            onClick={() => onCreate()}
          />
        }
        <table 
          className="table"
          style={{
            gridRow: "2",
            gridColumn: "1 / span 2",
            height: "100%",
            width: "100%",
          }}
        >
          <TableHeader
            columns={columns}
            sortColumns={columns}
            onChange={setSort}
            sort={sort}
          />
          <tbody>
            {rows.map((x, i) => {
              return (
                <tr key={i}>
                  <td>
                    {" "}
                    <a onClick={x.onClick}> {x.name} </a>{" "}
                  </td>
                  <td>
                    {" "}
                    <DateView value={x.createdAt} />{" "}
                  </td>
                  <td>
                    {" "}
                    <DeleteBtn onClick={x.onDelete} />{" "}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
  );
};
export default WorkspaceTable;
