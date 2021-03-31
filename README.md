# kaggle-moa

Competition description: https://www.kaggle.com/c/m5-forecasting-accuracy

Link to my kaggle profile: https://www.kaggle.com/ismailbbm

Summary: 5th edition of the famous M prediction [competitions](https://en.wikipedia.org/wiki/Makridakis_Competitions). This 5th edition goal is predicting items sales in a general store. The particularity being that zero sales occurences are quite high whether due to low demand or due to out of stock situations.

This solution achieved a top 2% score awarding a silver medal.

# Model

The model is based on a gradient boosting algorithm. I used lightgbm as it natively supports categorical variables without having to one-hot encode them, prior to feeding them to the model, which would have resulted in an overblown memory dataframe.
The model configuration itself is relatively simple - no ensembling or stacking for example. The only exotic features being the use of the competition metric for early stopping as well as a tweedie loss (since we have zero inflated targets).

The ideas which provide the biggest performance boosts described below are the following:
* Different models per week out
* Lag features especially on calendar events
* Zero sales due to zero demand vs out of stock inference
* Out of stock prediction

### One model per week out

The competition goal was to predict the sales 4 weeks in the future.
If we train only one gradient boosting algorithm to predict up to 28 days in the future, due to the design choices I made, I was not able to feed the model information that is less than 28 days old - also in inference. This means that when predicting the first day of the forecast period, the model can only be fed data that is older than 28 days in the past. 
Obviously this is inefficient as we do not use the most recent available information for prediction.

To circumvent this, ideally we would build 28 models for each out day of the prediction. One model to predict d+1, one for d+2... each one of these models would have then access to d+0 data. However this is very computing intensive and therefore experimenting is not very fast.
The compromise that I use is to build 4 models, the first one predicting the first week out...
This way, the first model would have access to data 7 days old, the second to data 14 days old...

### Lag features

The basic idea is that known future events can influence today's buyers behavior. For example, knowing that a big sporting event will happen over the week end can impact the sales of certain products during the week. Features are then introduced in the dataset to inform about future events.

### Zero demand vs out of stock

The target variable contains a lot of zeros. Upon inspection, some of these zeros seem to be due to out of stock issues rather than geniune no demand situation.
For example, if a product regularly sold in the hundreds of units daily and then from one day to the other, none were sold for few days before resuming again selling in the hundreds, we can reasonably assume that this is not due to the demand suddenly vanishing but rather an issue in the supply.

I have designed an algorithm which tries to differentiate between supply issues and geniune zero demand and used the boolean data as a feature in the model.

![out_of_stock.png](https://github.com/ismailbbm/kaggle-m5/blob/master/images/out_of_stock.png)

In the image above, the green line corresponds to the actual sales for a specific product. The red line corresponds to the out of stock boolean, when equal to one, it means that an out of stock issue is detected.

### Out of stock prediction

If we feed the model with the information about out of stock, we need to be able to handle the same during inference. I made the choice to handle that in a post processing step rather than in the modeling itself.
The idea is as follows:
* If the last known day has registered zero sales, we assume that it might be due to out of stock which will likely last in time.
* Depending on the historic for this product of how often out of stock issues happen and how long they last we compute two things: if the series of zero just preceding the forecast period is likely to be due to out of stock rather than zero demand and how long the out of stock issue is likely to last



