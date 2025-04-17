def set_template(args):
    # Set the templates here
    if args.template.find('dsmt') >= 0:
        args.input_setting = 'Y'
        args.input_mask = None
        args.scheduler = 'CosineAnnealingLR'
        args.max_epoch = 500
