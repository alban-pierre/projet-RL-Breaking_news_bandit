function res = plot_expectations(hot_tr, means, tmax)

    % Computes the expectation of an arm accross iterations

    % means : {N}(1*K)
    % alld  : (K*tmax)

    NbArms = size(means,2);
    
    for a=1:NbArms
        K = size(hot_tr{a},1);
        for k=1:K
            alld = zeros(K,tmax);
            pr = ((1:K)' == k);
            for j=1:tmax
                pr = hot_tr{a}*pr;
                alld(:,j) = pr;
            end
            res{k,a} = means{a}*alld;
        end
    end

    colors = ['b'; 'r'; 'k'; 'g'; 'm'; 'c'];
    types = ['-'; '--'; '.-'; '.'];
    
    figure;
    for a=1:NbArms
        for k=1:K
            plot(1:size(res{k,a},2), res{k,a}, [types(k,1), colors(a,1)]);
            hold on;
        end
    end

end
