function mu = knn(rw, tl, ta, k);

    k = min(k, size(tl,2)-1);
    
    X = [rw(1,1:end-1); tl(1,2:end)];
    y = [rw(1,end); ta];
    dd = sqdist(y,X);

    [s, si] = sort(dd);

    mu = mean(rw(1,1+si(1,1:k)),2);
    
end
