function [rew, draws] = KNN_UCB_NEW(tmax, MAB, squeezed)

    % Decides which arm to draw based on a k-nearest neightbors of the space (x = last reward of the arm, y = time between now and the last time this arm drawn)
    % -> we compute the mean of the k=sqrt(t) nearest neighbors of the current point in each arm, and choose the maximum of this mean + UCB variance
    % All points of the space are between 0 and (squeezed) for the reward axis, and between 0 and 1 for the time axis
    
    if (nargin < 3)
        squeezed = 1;
    end
    
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

    % Put rewards between 0 and (squeezed)
    rmin = min(rew(1,1:NbArms*2),[],2);
    rmax = max(rew(1,1:NbArms*2),[],2);
    for i=1:NbArms
        rw{i} = squeezed*(rw{i} - rmin)./(rmax - rmin);
    end
    
    for t=2*NbArms+1:tmax
%        if (rand(1) < 0.1) % To force exploration
%            ima = randi(NbArms);
%        else
            for i=1:NbArms
                % k nearest neightbors algo
                % Here the rewards are bounded by 0 and (squeezed), which was is not the case of UCB_KNN_OLD where the resize is made just after the knn computation
                mu(1,i) = knn(rw{i}, 2*exp(-tl{i}+1), 2*exp(-ta(1,i)+1), ceil(na(1,i)/700)*ceil(sqrt(na(1,i))));%ceil(sqrt(t/NbArms)));
            end
            %mu = (mu - rmin)./(rmax - rmin);
            [~, ima] = max(mu./squeezed + sqrt(log(t)./(2*na)), [], 2); % arm chosen given the expectation and the variance
%        end
        rew(1,t) = MAB.sample(ima);
        % Update bounds if necessary
        if (rew(1,t) > rmax)
            for i=1:NbArms
                rw{i} = rw{i}./((rew(1,t)-rmin)./(rmax-rmin));
            end
            rmax = rew(1,t);
        elseif (rew(1,t) < rmin)
            for i=1:NbArms
                rw{i} = squeezed - (squeezed - rw{i})./((rmax-rew(1,t))./(rmax-rmin));
            end
            rmin = rew(1,t);
        end
        rw{ima} = [rw{ima}, squeezed*(rew(1,t) - rmin)/(rmax - rmin)];
        tl{ima} = [tl{ima}, ta(1,ima)];
        ta(1,ima) = 0;
        ta = ta+1;
        %smu(1,ima)+=rew(1,t);
        na(1,ima) = na(1,ima) + 1;
        %mu(1,ima) = smu(1,ima)/na(1,ima);
    end
    draws = na;
end
