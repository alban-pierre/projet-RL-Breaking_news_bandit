function [rew, draws, hot_expected, hot_real] = UCB_BN(tmax, MAB)
    %UCB for Breaking News bandit (one hot arm)
    % Does not work yet since rewards are not bounded by [0,1]
    % Update : I used min and max of rewards in order to put rewards artificially between 0 and 1

    
    NbArms=MAB.nbArms();

    tmax = max(tmax, NbArms);

    mu = zeros(1,NbArms);
    smu = zeros(1,NbArms);
    max_rew = zeros(1,NbArms);
    na = ones(1,NbArms);
    rew = zeros(1,tmax);
    
    %Vectors of real hot state and expected hot state
    hot_expected = [];
    hot_real = [];
    
    %disp('----------------init');
    hotState = 0; %Indicator of the state that is supposed hot
    old_na_hot = 0; %Hold the number of pulls before the state became hot
    old_max_rew = 0;
    %old_mu_hot = 0;
    old_smu_hot = 0;
    
    for i=1:NbArms
        [mu(1,i), h] = MAB.sample(i);
        max_rew(1,i) = mu(1,i);
        
        hot_expected = [hot_expected, hotState];
        hot_real = [hot_real, h];
    end
    rew(1,1:NbArms) = mu;
    smu = mu;

    rmin = min(rew,[],2);
    rmax = max(rew,[],2);

    mu = ((smu./na) - rmin)./(rmax - rmin);
    
    for t=NbArms+1:tmax
        if (hotState == 0) %If no arm is supposed hot
            [ma, ima] = max(mu+sqrt(log(t)./(2*na)), [], 2);
            [rew(1,t), h] = MAB.sample(ima);
            if ((rew(1,t)-rmin)/(rmax-rmin) <= (max_rew(1,ima)-rmin)/(rmax-rmin)+sqrt(log(t)./(2*na(1,ima)))) %The sample is in the range expected: arm not hot
                smu(1,ima) = smu(1,ima) + rew(1,t);
                max_rew(1,ima) = max(max_rew(1,ima), rew(1,t));
                na(1,ima) = na(1,ima) + 1;
                if (rew(1,t) > rmax)
                    rmax = rew(1,t);
                    mu = ((smu./na) - rmin)./(rmax - rmin);
                elseif (rew(1,t) < rmin)
                    rmin = rew(1,t);
                    mu = ((smu./na) - rmin)./(rmax - rmin);
                else
                    mu(1,ima) = ((smu(1,ima)/na(1,ima)) - rmin)/(rmax - rmin);
                end
            else %The sampled reward is over the expected range: we suppose it is hot
                hotState = ima;
                %disp(['Hot state detected ', num2str(ima), ' at time ', num2str(t)]);
                %Hold the values before entering hot state
                old_na_hot = na(1,ima);
                old_max_rew = max_rew(1,ima);
                old_smu_hot = smu(1,ima);
                %Enter hot state with new values
                smu(1,ima) = rew(1,t);
                na(1,ima) = 1;
                max_rew(1,ima) = rew(1,t)+(rmin+(rmax-rmin)*sqrt(log(t)./(2*na(1,ima))));
                %Normalize as usual if needed
                if (rew(1,t) > rmax)
                    rmax = rew(1,t);
                    mu = ((smu./na) - rmin)./(rmax - rmin);
                elseif (rew(1,t) < rmin)
                    rmin = rew(1,t);
                    mu = ((smu./na) - rmin)./(rmax - rmin);
                else
                    mu(1,ima) = ((smu(1,ima)/na(1,ima)) - rmin)/(rmax - rmin);
                end
            end
            hot_expected = [hot_expected, hotState];
            hot_real = [hot_real, h];
        else
            [ma, ima] = max(mu+sqrt(log(t)./(2*na)), [], 2);
            [rew(1,t),h] = MAB.sample(ima);
            if (ima == hotState && (rew(1,t)-rmin)/(rmax-rmin) < (max_rew(1,ima)-rmin)/(rmax-rmin)-sqrt(log(t)./(2*na(1,ima)))) % Leave hot state
                %disp(['Leaving hot state detected ', num2str(ima), ' at time ', num2str(t)]);
                hotState = 0;
                %Take back old values
                na(1,ima) = old_na_hot + 1;
                smu(1,ima) = old_smu_hot + rew(1,t);
                max_rew(1,ima) = old_max_rew;
                %Normalize if needed
                if (rew(1,t) > rmax)
                    rmax = rew(1,t);
                    mu = ((smu./na) - rmin)./(rmax - rmin);
                elseif (rew(1,t) < rmin)
                    rmin = rew(1,t);
                    mu = ((smu./na) - rmin)./(rmax - rmin);
                else
                    mu(1,ima) = ((smu(1,ima)/na(1,ima)) - rmin)/(rmax - rmin);
                end
            else %Don't change hot state
                smu(1,ima) = smu(1,ima) + rew(1,t);
                na(1,ima) = na(1,ima) + 1;
                if (ima == hotState)
                    %We actually hold the min reward for the hot state to
                    %check if below the min expected value
                    max_rew(1,ima) = min(max_rew(1,ima), rew(1,t)+(rmin+(rmax-rmin)*sqrt(log(t)./(2*na(1,ima)))));
                    %disp(['var: ', num2str((rmin+(rmax-rmin)*sqrt(log(t)./(2*na(1,ima)))))]);
                    %disp(['rew: ', num2str(rew(1,t))]);
                    %disp(['max_rew: ', num2str(max_rew(1,ima))]);
                else
                    max_rew(1,ima) = max(max_rew(1,ima), rew(1,t));
                end
                if (rew(1,t) > rmax)
                    rmax = rew(1,t);
                    mu = ((smu./na) - rmin)./(rmax - rmin);
                elseif (rew(1,t) < rmin)
                    rmin = rew(1,t);
                    mu = ((smu./na) - rmin)./(rmax - rmin);
                else
                    mu(1,ima) = ((smu(1,ima)/na(1,ima)) - rmin)/(rmax - rmin);
                end
            end
            hot_expected = [hot_expected, hotState];
            hot_real = [hot_real, h];
        end
    end
    draws = na;
end
