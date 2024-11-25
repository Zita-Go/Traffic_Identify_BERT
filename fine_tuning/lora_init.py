import loralib as lora


def get_lora_bert_model(model, r=8, lora_layer=["q", 'k', 'v', 'o']):
    encoder_layers = list(model.encoder.transformer)
    for layer_index, encoder_layer in enumerate(encoder_layers):
        # 访问多头自注意力层
        attention = encoder_layer.self_attn
        # 获取Q、K、V线性层
        layers = list(attention.linear_layers)
        q_linear = layers[0]
        k_linear = layers[1]
        v_linear = layers[2]
        # 获取O线性层（实际上，O是V经过加权求和后的结果，通常不单独存储）
        o_linear = attention.final_linear

        for l in lora_layer:
            if l == 'q':
                new_q_proj = lora.Linear(q_linear.in_features, q_linear.out_features, r=r, lora_alpha=2*r)
                model.encoder.transformer[layer_index].self_attn.linear_layers[0] = new_q_proj
            elif l == 'k':
                new_k_proj = lora.Linear(k_linear.in_features, k_linear.out_features, r=r, lora_alpha=2*r)
                model.encoder.transformer[layer_index].self_attn.linear_layers[1] = new_k_proj
            elif l == 'v':
                new_v_proj = lora.Linear(v_linear.in_features, v_linear.out_features, r=r, lora_alpha=2*r)
                model.encoder.transformer[layer_index].self_attn.linear_layers[2] = new_v_proj
            elif l == 'o':
                new_o_proj = lora.Linear(o_linear.in_features, o_linear.out_features, r=r, lora_alpha=2*r)
                model.encoder.transformer[layer_index].self_attn.final_linear = new_o_proj
    output_1 = model.output_layer_1
    output_2 = model.output_layer_2
    model.output_layer_1 = lora.Linear(output_1.in_features, output_1.out_features, r=r, lora_alpha=20*r)
    model.output_layer_2 = lora.Linear(output_2.in_features, output_2.out_features, r=r, lora_alpha=20*r)
    return model

