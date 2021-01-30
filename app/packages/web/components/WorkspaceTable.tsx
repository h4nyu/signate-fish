import React, { useState } from 'react';
import { List } from "immutable";

import { Workspace } from "@sivic/core/workspace";
import DateView from "@sivic/web/components/DateView";
import TableHeader from "@sivic/web/components/TableHeader";

const columns = [
  "Name",
  "Create",
];

const filterColumns = [
  "Name",
];

export const WorkspaceTable = (props: {
  workspaces: Workspace[];
  onClick?: (id: string) => void;
  onAdd?: (keyword: string) => void;
}) => {
  const { workspaces, onClick , onAdd} = props;
  const [sort, setSort] = React.useState<[string, boolean]>(["Name", true]);
  const [sortColumn, asc] = sort;
  const [keyword, setKeyword] = useState("");
  const lowerKeyowerd = keyword.toLowerCase();

  let rows = List(workspaces).map(x => {
    return {
      ...x,
      Name: x.name,
      Create:x.createdAt,
      onClick: () => onClick && onClick(x.id),
    }
  })
  .filter(x =>  filterColumns
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
    <div style={{width:"100%"}}>
      <div style={{
        display: "flex",
        flexDirection: 'row'
        }}
      >
        <input
          className="input"
          type="text"
          onChange={(e) => setKeyword(e.target.value)}
        />
        <button className={"button is-light"} onClick={() => onAdd && onAdd(keyword)}>
          Add 
        </button>
      </div>
      <table className="table is-fullwidth">
        <TableHeader
          columns={columns}
          sortColumns={columns}
          onChange={setSort}
          sort={sort}
        />
        <tbody>
          {rows
            .map((x, i) => {
              return (
                <tr
                  key={i}
                  style={{ cursor: onClick ? "pointer" : "" }}
                >
                  <td> <a onClick={x.onClick}> {x.name} </a> </td>
                  <td> <DateView value={x.createdAt} /> </td>
                </tr>
              );
            })}
        </tbody>
      </table>
    </div>
  );
};
export default WorkspaceTable;