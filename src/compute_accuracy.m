function [res, bestThresh] = compute_accuracy(scores, ground_truth)
    
    % finds an optimal threshold - the threshold which maximises the accuracy

    % threshold scores and get the sign
        
    res = -Inf;
    bestThresh = [];
    
    % thresh loop
    for i=1:numel(scores)
         
        curThresh = scores(i);
        class = 2 * (scores >= curThresh) - 1;
        
        % class-n accuracy
        acc = mean(class == ground_truth);
        
        if acc > res
            
            res = acc;
            bestThresh = curThresh;            
        end
    end
    
    res = res * 100;
    
end
