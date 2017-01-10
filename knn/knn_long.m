function amu = knn_long(rew, airw, arw, atl, ata, ana);

%   mu(1,i) = knn_long(rw{i}, 2*exp(-tl{i}+1), 2*exp(-ta(1,i)+1), ceil(na(1,i)/700)*ceil(sqrt(na(1,i))), ceil(na(1,i)/500));%ceil(sqrt(t/NbArms)));

    NbArms = size(ana,2);
    amu = zeros(1,NbArms);
    
    for i=1:NbArms

        rw = arw{i};
        irw = airw{i};
        tl = 2*exp(-atl{i}+1);
        ta = 2*exp(-ata(1,i)+1);
        k = ceil(ana(1,i)/700)*ceil(sqrt(ana(1,i)));
        l = ceil(ana(1,i)/500);
        
        k = min(k, size(tl,2)-1);
        
        X = [rw(1,1:end-l); tl(1,2:end-l+1)];
        y = [rw(1,end); ta];
        dd = sqdist(y,X);

        mu = 0;
        [s, si] = sort(dd);
        sl = [];
        for li=0:l-1
            sl = [sl, 0.8.^li];
            %mu = mu + sl(1,end)*mean(rw(1,li+si(1,1:k)),2);
            mu = mu + sl(1,end)*mean(rew(1,li+irw(1,1+si(1,1:k))),2);
        end
        amu(1,i) = mu/sum(sl,2);

    end
    
end
