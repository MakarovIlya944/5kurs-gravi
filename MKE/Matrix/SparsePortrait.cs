using System;
using System.Collections.Generic;
using System.Linq;

namespace MKE.Matrix
{
    public class SparsePortrait
    {
        public Dictionary<int, HashSet<int>> Rows { get; } = new Dictionary<int, HashSet<int>>();
        public HashSet<int> Columns { get; } = new HashSet<int>();

        public IEnumerable<int> RowsIndices => Rows.Keys.OrderBy(i => i);
        public IEnumerable<int> ColumnIndices => Columns.OrderBy(i => i);
        public int[] PremutationRowIndices { get; private set; }
        public int[] PremutationColumnIndices { get; private set; }

        public void Add(IEnumerable<int> rows, IEnumerable<int> columns)
        {
            foreach (var item in rows)
            {
                if (!Rows.TryGetValue(item, out var row))
                    Rows[item] = row = new HashSet<int>();
                row.UnionWith(columns);
            }
            Columns.UnionWith(columns);
        }

        public void Add(ReadOnlySpan<int> rows, ReadOnlySpan<int> columns)
        {
            var colArray = columns.ToArray();
            foreach (var item in rows)
            {
                if (!Rows.TryGetValue(item, out var row))
                    Rows[item] = row = new HashSet<int>();
                row.UnionWith(colArray);
            }
            Columns.UnionWith(colArray);
        }

        public void DeleteRows(IList<int> rows)
        {
            if (!rows.Any()) return;
            foreach (var row in rows)
                Rows.Remove(row);
        }

        public void DeleteColumns(IList<int> columns)
        {
            if (!columns.Any()) return;
            bool[] flags = new bool[Math.Max(columns.Max(), Columns.Max()) + 1];
            for (int i = 0; i < columns.Count; i++)
                flags[columns[i]] = true;

            foreach (var row in Rows)
                row.Value.RemoveWhere(i => flags[i]);
            Columns.ExceptWith(columns);
        }

        public void DeleteDofs(bool[] mask)
        {
            var deletedRows = Rows.Keys.Where(x => mask[x]).ToList();
            foreach (var row in deletedRows)
                Rows.Remove(row);

            foreach (var row in Rows)
                row.Value.RemoveWhere(x => mask[x]);
            Columns.RemoveWhere(x => mask[x]);
        }

        public IList<IEnumerable<int>> GetLinks()
        {
            var rowsList = new IEnumerable<int>[Rows.Keys.Max() + 1];
            foreach (var item in Rows)
                rowsList[item.Key] = item.Value.OrderBy(i => i).ToList();
            return rowsList;
        }

        public IList<IEnumerable<int>> GetMappedLinks()
        {
            var rowsPerm = RowsIndices.ToArray();
            var colsPerm = ColumnIndices.ToArray();

            var rowsIPerm = new int[rowsPerm.LastOrDefault() + 1];
            for (int i = 0; i < rowsPerm.Length; i++)
                rowsIPerm[rowsPerm[i]] = i;

            var colsIPerm = new int[colsPerm.LastOrDefault() + 1];
            for (int i = 0; i < colsPerm.Length; i++)
                colsIPerm[colsPerm[i]] = i;

            PremutationRowIndices = rowsIPerm;
            PremutationColumnIndices = colsIPerm;
            var rowsList = new IEnumerable<int>[rowsPerm.Length];
            foreach (var item in Rows)
            {
                rowsList[rowsIPerm[item.Key]] = item.Value.Select(i => colsIPerm[i]).OrderBy(i => i).ToArray();
            }
            return rowsList;
        }

        public SparsePortrait Permutation(int[] perm)
        {
            var res = new SparsePortrait();
            foreach (var col in Columns)
            {
                res.Columns.Add(perm[col]);
            }
            foreach (var row in Rows)
            {
                var newrow = new HashSet<int>();
                foreach (var col in row.Value)
                {
                    newrow.Add(perm[col]);
                }
                res.Rows.Add(perm[row.Key], newrow);
            }
            return res;
        }
    }
}