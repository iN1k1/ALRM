function [n] = NM_norm(w, type)
switch lower(type)
    case {'1', '2'}
        n = norm(w, str2num(type));
    case 'fro'
        n = norm(w, 'fro');
    case '21'
        n = sum(sqrt(sum(w.*w,2)));
end
    
end