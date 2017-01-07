function img = plot_knn_points(rw, tl)

    T = size(tl,2);

    img = zeros(T);
    na = zeros(T);
    
    mi = min(rw);
    ma = max(rw);

    indice = @(r) (ceil((r-mi)/(ma-mi)*(T-1)+0.5));

    X = [rw(1,1:end-1); tl(1,2:end)];
    
    for i=1:size(tl,2)-1
        img(X(2,i), indice(X(1,i))) = img(X(2,i), indice(X(1,i))) + rw(1,i+1);
        na(X(2,i), indice(X(1,i))) = na(X(2,i), indice(X(1,i))) + 1;
    end

    na = max(na,1);

    imag = img ./ na;

    figure;
    imagesc(log(img+1));

end
