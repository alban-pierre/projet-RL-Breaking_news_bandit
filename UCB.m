function [rew, draws] = UCB(tmax, MAB)

    NbArms=length(MAB);

    tmax = max(tmax, NbArms);

    mu = zeros(1,NbArms);
    smu = zeros(1,NbArms);
    na = ones(1,NbArms);
    rew = zeros(1,tmax);
    
    for i=1:NbArms
	mu(1,i) = sample(MAB{i});
    end
    rew(1,1:NbArms) = mu;
    smu = mu;
    
    for t=NbArms+1:tmax
	[ma, ima] = max(mu+sqrt(log(t)./(2*na)), [], 2);
	rew(1,t) = sample(MAB{ima});
	smu(1,ima)+=rew(1,t);
	na(1,ima)++;
	mu(1,ima) = smu(1,ima)/na(1,ima);
    end
    draws = na;
end
