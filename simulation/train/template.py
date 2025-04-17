def set_template(args):
    # Set the templates here
    if args.template.find('dsmt') >= 0:
        args.input_setting = 'H'
        args.input_mask = 'Mask'
        args.scheduler = 'CosineAnnealingLR'
        args.max_epoch = 500
