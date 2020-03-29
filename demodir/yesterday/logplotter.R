require(data.table)
require(dplyr)
require(tidyr)
require(ggplot2)
getwd()
train<-fread("csvs/train.csv")
valid<-fread('csvs/valid.csv')
colnames(train) <- make.names(colnames(train), unique=TRUE)
colnames(valid) <- make.names(colnames(valid), unique=TRUE)

train<-train %>% mutate(train.epoch=train.epoch+resampling/200)

# value
train.value<-train %>% select(epoch=train.epoch, value.loss=running.value.loss)
beh<-rep("Training",nrow(train.value))
train.value<-train.value %>% mutate("Dataset"=beh)

valid.value<-valid %>% select(epoch=valid.epoch, value.loss=validation.value.loss)
meh<-rep("Validation",nrow(valid.value))
valid.value<-valid.value %>% mutate("Dataset"=meh)

value<-rbind(train.value, valid.value)

value.plot<-ggplot()+
  geom_line(data=value, aes(x=epoch, y=value.loss, color=Dataset))+
  labs(x="Epoch", y="Value loss")+ theme(legend.position="bottom")
ggsave("plots/value.png", width=4, height=3)

# policy
train.policy<-train %>% select(epoch=train.epoch, policy.loss=running.policy.loss)
beh<-rep("Training",nrow(train.policy))
train.policy<-train.policy %>% mutate("Dataset"=beh)

valid.policy<-valid %>% select(epoch=valid.epoch, policy.loss=validation.policy.loss)
meh<-rep("Validation",nrow(valid.policy))
valid.policy<-valid.policy %>% mutate("Dataset"=meh)

policy<-rbind(train.policy, valid.policy)

policy.plot<-ggplot()+
  geom_line(data=policy, aes(x=epoch, y=policy.loss, color=Dataset))+
  labs(x="Epoch", y="Policy loss")+ theme(legend.position="bottom")
ggsave("plots/policy.png", width=4, height=3)

# pdiff
train.pdiff<-train %>% select(epoch=train.epoch, pdiff=running.p.diff)
beh<-rep("Training",nrow(train.pdiff))
train.pdiff<-train.pdiff %>% mutate("Dataset"=beh)

valid.pdiff<-valid %>% select(epoch=valid.epoch, pdiff=validation.p.diff)
meh<-rep("Validation",nrow(valid.pdiff))
valid.pdiff<-valid.pdiff %>% mutate("Dataset"=meh)

pdiff<-rbind(train.pdiff, valid.pdiff)

pdiff.plot<-ggplot()+
  geom_line(data=pdiff, aes(x=epoch, y=pdiff, color=Dataset))+
  labs(x="Epoch", y="Mean policy L1 distance")+ theme(legend.position="bottom")

ggsave("plots/pdiff.png", width=4, height=3)

# value and policy

