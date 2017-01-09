function mu = nnd(rw, tl, ta, d);

    X = [rw(1,1:end-1); tl(1,2:end)];
    y = [rw(1,end); ta];
    dd = sqdist(y,X);

    si = dd < d.^2;

    if (sum(si,2) > 0)
        mu = mean(rw(1,si),2);
    else
        [~, imu] = min(dd,[],2);
        mu = rw(1,imu);
    end
end
