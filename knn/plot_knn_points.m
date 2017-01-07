function img = plot_knn_points(rw, tl)

    T = 100;

    mi2 = min(tl);
    ma2 = max(tl);
    
    img = zeros(T);
    na = zeros(T);
    
    mi = min(rw);
    ma = max(rw);

    indice = @(r) (ceil((r-mi)/(ma-mi)*(T-1)+0.5));
    indice2 = @(r) (ceil((r-mi2)/(ma2-mi2)*(T-1)+0.5));

    X = [rw(1,1:end-1); tl(1,2:end)];
    
    for i=1:size(tl,2)-1
        img(indice2(X(2,i)), indice(X(1,i))) = img(indice2(X(2,i)), indice(X(1,i))) + rw(1,i+1);
        na(indice2(X(2,i)), indice(X(1,i))) = na(indice2(X(2,i)), indice(X(1,i))) + 1;
    end

    na = max(na,1);

    img = img ./ na;

    figure;
    imagesc(log(img-mi+1));

end
