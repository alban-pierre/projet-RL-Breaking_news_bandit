function [rew, draws] = KNN_UCB_OLD(tmax, MAB)

    % Decides which arm to draw based on a k-nearest neightbors of the space (x = last reward of the arm, y = time between now and the last time this arm drawn)

    NbArms=MAB.nbArms();

    tmax = max(tmax, 2*NbArms);

    %mu = zeros(1,NbArms);
    %smu = zeros(1,NbArms);
    na = 2*ones(1,NbArms);
    rew = zeros(1,tmax);
    ta = ones(1,NbArms);
    mu = ones(1,NbArms);

    clear rw;
    clear tl;
    for i=1:NbArms
        rw{i} = [];
        tl{i} = [];
    end

    for j=0:1
        for i=1:NbArms
            rew(1,j*NbArms+i) = MAB.sample(i);
            rw{i} = [rw{i}, rew(1,j*NbArms+i)];
            tl{i} = [tl{i}, ta(1,i)];
            ta(1,i) = 0;
            ta = ta+1;
        end
    end

    rmin = min(rew,[],2);
    rmax = max(rew,[],2);
    
    for t=2*NbArms+1:tmax
%        if (rand(1) < 0.1) % To force exploration
%            ima = randi(NbArms);
%        else
            for i=1:NbArms
                mu(1,i) = knn(rw{i}, 2*exp(-tl{i}+1), 2*exp(-ta(1,i)+1), ceil(sqrt(t/NbArms)));
            end
            mu = (mu - rmin)./(rmax - rmin);
            [ma, ima] = max(mu+sqrt(log(t)./(2*na)), [], 2);
%        end
        rew(1,t) = MAB.sample(ima);
        rmax = max(rmax, rew(1,t));
        rmin = min(rmin, rew(1,t));
        rw{ima} = [rw{ima}, rew(1,t)];
        tl{ima} = [tl{ima}, ta(1,ima)];
        ta(1,ima) = 0;
        ta = ta+1;
        %smu(1,ima)+=rew(1,t);
        na(1,ima) = na(1,ima) + 1;
        %mu(1,ima) = smu(1,ima)/na(1,ima);
    end
    draws = na;
end
