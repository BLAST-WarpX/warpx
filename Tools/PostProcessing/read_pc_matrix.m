function [A, info, dofmap] = read_pc_matrix(filename)
% READ_PC_MATRIX  Read a preconditioner matrix written by WarpX
%                 with <prefix>.print_matrix = 1 (MATLAB and Octave).
%
%   [A, info, dofmap] = read_pc_matrix('diags/pc_matrices/pc_jacobi_step00000_update0.mtx')
%
%   A       sparse matrix (global DOF indices)
%   info    struct of the "% key: value" metadata lines (values are strings)
%   dofmap  struct with fields lev, comp (0: x, 1: y, 2: z), i, j, k, indexed
%           by row; read from dofmap.txt in the same directory
%
% Copyright 2026 The WarpX Community
% This file is part of WarpX. License: BSD-3-Clause-LBNL

fid = fopen(filename, 'r');
if fid < 0
    error('read_pc_matrix: cannot open %s', filename);
end
info = struct();
line = fgetl(fid);
while ischar(line) && ~isempty(line) && line(1) == '%'
    if numel(line) < 2 || line(2) ~= '%'
        pos = strfind(line, ':');
        if ~isempty(pos)
            key = strtrim(line(2:pos(1)-1));
            key = regexprep(key, '[^A-Za-z0-9_]', '_');
            info.(key) = strtrim(line(pos(1)+1:end));
        end
    end
    line = fgetl(fid);
end
sizes = sscanf(line, '%d %d %d');
data = textscan(fid, '%f %f %f');
fclose(fid);
A = sparse(data{1}, data{2}, data{3}, sizes(1), sizes(2));

if nargout > 2
    dofmap_file = fullfile(fileparts(filename), 'dofmap.txt');
    fid = fopen(dofmap_file, 'r');
    if fid < 0
        error('read_pc_matrix: cannot open %s', dofmap_file);
    end
    d = textscan(fid, '%f %f %f %f %f %f', 'CommentStyle', '#');
    fclose(fid);
    [~, order] = sort(d{1});
    names = {'lev', 'comp', 'i', 'j', 'k'};
    dofmap = struct();
    for n = 1:numel(names)
        dofmap.(names{n}) = d{n+1}(order);
    end
end
end
