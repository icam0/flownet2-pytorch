
class ModelAndLoss(nn.Module):
    def __init__(self, args):
        super(ModelAndLoss, self).__init__()
        kwargs = tools.kwargs_from_args(args, 'model')
        self.model = args.model_class(args, **kwargs)
        kwargs = tools.kwargs_from_args(args, 'loss')
        self.loss = args.loss_class(args, **kwargs)

    def forward(self, data, target, inference=False ):
        output = self.model(data)

        loss_values = self.loss(output, target)

        if not inference :
            return loss_values
        else :
            return loss_values, output

model_and_loss = ModelAndLoss(args)