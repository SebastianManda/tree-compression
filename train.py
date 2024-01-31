import numpy as np
import nvdiffrast.torch as dr
import matplotlib
import random
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import torch.nn.functional as F
import torch
from torchvision import transforms
from PIL import Image
import os
import image_similarity_measures.quality_metrics as qm


RES = 1024


def fetch_ground_truth_images(name):
    convert_tensor = transforms.ToTensor()

    file_name = "data/" + name + "_ground_truth"
    all_images = os.listdir(file_name)
    all_images_indexes = list(map(lambda x: int(float(x.split("_")[2])), all_images))
    zipped_list = list(zip(all_images_indexes, all_images))
    zipped_list.sort(key=lambda x: x[0])
    images = list(map(lambda x: x[1], zipped_list))

    ground_truth = []
    for img_name in images:
        img = Image.open(file_name + "/" + img_name)
        img_tensor = convert_tensor(img)
        final = img_tensor.permute(1, 2, 0)
        ground_truth.append(final)

    return torch.stack(ground_truth)


def save_image(content, path):
    numpy_image = (content * 255.0).clamp(0, 255).to(torch.uint8).cpu().detach().numpy()  # Scale to 0-255 and convert to uint8
    pil_image = Image.fromarray(numpy_image)  # Transpose dimensions for PIL
    pil_image.save(path + ".png")


def save_comparison_image(optimized, ground_truth, path):
    image_np = (optimized * 255).clamp(0, 255).byte().cpu().numpy()
    image_np_gt = (ground_truth * 255).clamp(0, 255).byte().cpu().numpy()
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(image_np)
    axarr[0].set_title('Optimized')
    axarr[1].imshow(image_np_gt)
    axarr[1].set_title('Ground Truth')
    plt.savefig(path)
    plt.close()


def rotate_tensor_y(tensor, angle):
    angle = np.radians(angle)
    s, c = np.sin(angle), np.cos(angle)
    r_mtx = np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c],]).astype(np.float32)
    r_tensor = torch.tensor(r_mtx, dtype=torch.float32).cuda()
    return torch.matmul(tensor.clone().cuda(), r_tensor.t())



def displace_in_normal_dir(tensor):
    return tensor.cuda() + torch.tensor([0.0, 0.0, 1.0]).cuda() * (-0.00001)


class Plane:
    def __init__(self, angle, bounds):
        self.angle = angle
        self.bounds = bounds

        pos_data = [
            [bounds[0], bounds[3], 0.0],
            [bounds[1], bounds[3], 0.0],
            [bounds[0], bounds[2], 0.0],
            [bounds[1], bounds[2], 0.0]
        ]
        self.pos = torch.tensor(pos_data, dtype=torch.float32).cuda()
        self.pos = displace_in_normal_dir(self.pos)
        self.pos = rotate_tensor_y(self.pos, angle)
        self.pos_idx = torch.tensor([[0, 1, 3], [0, 3, 2]], dtype=torch.int32).cuda()

        bnd_res_x = RES * (bounds[1] - bounds[0]) / 2
        bnd_res_y = RES * (bounds[3] - bounds[2]) / 2
        self.tex = (int(bnd_res_x), int(bnd_res_y), 4)
        self.vtx_uv = torch.tensor([
            [0.0, bnd_res_y * 1.0],
            [bnd_res_x * 1.0, bnd_res_y * 1.0],
            [0.0, 0.0],
            [bnd_res_x * 1.0, 0.0]
        ])


def merge_planes(planes, angle=0, filter=False, filter_angle=45):
    pos = []
    pos_idx = []
    vtx_uv = []
    tex = (RES, 0, planes[0].tex[2])

    entire_tex_size = 0
    for plane in planes:
        is_within_view_angle = not filter or within_view_angle(plane.angle, angle, filter_angle=filter_angle)
        if is_within_view_angle:
            current_vtx_uv = plane.vtx_uv.clone()
            current_vtx_uv[:, 0] += entire_tex_size
            vtx_uv.append(current_vtx_uv)

            tex = (tex[0], tex[1] + plane.tex[1], tex[2])
            entire_tex_size += plane.tex[1]

            pos_idx.append(plane.pos_idx + len(pos) * 4)
            pos.append(plane.pos)

        else:
            entire_tex_size += plane.tex[1]

    if len(vtx_uv) > 0:
        vtx_uv = torch.cat(vtx_uv)
        vtx_uv[:, 0] /= entire_tex_size
        vtx_uv[:, 1] /= tex[0]
    else:
        vtx_uv = torch.tensor([], dtype=torch.float32).cuda()

    return torch.cat(pos).cuda().contiguous(), torch.cat(pos_idx).cuda().contiguous(), vtx_uv.cuda().contiguous(), torch.zeros(tex).cuda().contiguous()


def within_view_angle(plane_angle, angle, filter_angle):
    after_rotation_angle = (plane_angle + angle) % 360
    return after_rotation_angle > (360 - filter_angle) or after_rotation_angle < (0 + filter_angle)


def get_pos_clip_rotated(pos, angle):
    rotated_pos = rotate_tensor_y(pos, angle)
    return torch.cat([rotated_pos.cuda(), torch.ones([rotated_pos.shape[0], 1]).cuda()], dim=1).cuda()


def create_out_dir(dir_name, custom_name):
    output_dir = dir_name + "/" + datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + "_" + custom_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(output_dir + "/textures")
        os.makedirs(output_dir + "/renders")
    return output_dir


def get_image_bounds(image, transparency):
    non_black_mask = torch.any(image[:, :, :3] > 0, dim=2)
    if transparency:
        non_black_mask = (image[:, :, 3] > 0.0)
    non_black_indices = torch.nonzero(non_black_mask, as_tuple=False)

    min_x = (torch.min(non_black_indices[:, 1]) - (RES/2)) / (RES/2)
    max_x = (torch.max(non_black_indices[:, 1]) - (RES/2)) / (RES/2)
    min_y = (torch.min(non_black_indices[:, 0]) - (RES/2)) / (RES/2)
    max_y = (torch.max(non_black_indices[:, 0]) - (RES/2)) / (RES/2)

    return [min_x.item(), max_x.item(), min_y.item(), max_y.item()]


def create_bounded_plane_from_image(gnd_truth, angle, transparency=False):
    gnd_sample_coefficient = 360 / gnd_truth.shape[0]
    color_gnd = gnd_truth[int(((360 - angle) % 360) / gnd_sample_coefficient)]
    img_bounds = get_image_bounds(color_gnd, transparency)
    return Plane(angle, img_bounds)


def save_model(pos, pos_idx, vtx_uv, path):
    with open(path + "model.obj", "w") as obj_file:
        for v, uv in zip(pos, vtx_uv):
            obj_file.write(f"v {v[0]} {v[1]} {v[2]}\n")
            obj_file.write(f"vt {uv[0]} {uv[1]}\n")

        for f in pos_idx:
            obj_file.write(f"f {f[0] + 1}/{f[0] + 1} {f[1] + 1}/{f[1] + 1} {f[2] + 1}/{f[2] + 1}\n")


def alpha_blend(background, foreground):
    alpha_foreground = foreground[:, :, 3:4]
    alpha_background = background[:, :, 3:4]

    blended_alpha = alpha_foreground + alpha_background * (1 - alpha_foreground)
    blended_image = (alpha_foreground * foreground[:, :, :3] + alpha_background * background[:, :, :3] * (1 - alpha_foreground))

    return torch.cat([blended_image, blended_alpha], dim=2)


def render(pos, angle, cuda_ctx, pos_idx, vtx_uv, tex_opt, ranges, depth_layers):
    color_layers = []
    pos_clip = get_pos_clip_rotated(pos, angle)
    with dr.DepthPeeler(cuda_ctx, pos_clip, pos_idx, (RES, RES), ranges) as peeler:
        for k in range(depth_layers):
            rast_out, _ = peeler.rasterize_next_layer()
            texc, _ = dr.interpolate(vtx_uv, rast_out, pos_idx)
            color_opt = dr.texture(tex_opt[None, ...], texc, filter_mode='linear')
            color_opt = torch.where(rast_out[..., 3:] > 0, color_opt, torch.zeros(4).cuda())
            color_layers.append(color_opt)

    color_layers_reversed = list(reversed(color_layers))
    background = torch.zeros((RES, RES, 4)).cuda()
    for layer in color_layers_reversed:
        background = alpha_blend(background, layer[0])

    return background


def run(MAX_ITER=10000, SAVE_INTERVAL=100, plane_angles=[0, 45, 90, 135, 180, 225, 270, 315], filter_planes=False, filter_angle=45, evaluate=False, bounded=False, name="", L1=False, gnd_name="aspen"):

    gnd_truth = fetch_ground_truth_images(gnd_name)
    cuda_ctx = dr.RasterizeCudaContext()

    # Generate initial model, texture and necessary data
    planes = []
    for a in plane_angles:
        if bounded:
            planes.append(create_bounded_plane_from_image(gnd_truth, a, transparency=True))
        else:
            planes.append(Plane(a, [-1.0, 1.0, -1.0, 1.0]))

    pos, pos_idx, vtx_uv, tex = merge_planes(planes)
    ranges = torch.tensor([[0, pos_idx.shape[0]]], dtype=torch.int32)
    ranges = ranges.expand((1, -1)).contiguous()

    tex_data = torch.zeros(tex.shape, dtype=torch.float32).cuda()
    tex_opt = torch.tensor(tex_data, dtype=torch.float32, device='cuda', requires_grad=True)
    optimizer = torch.optim.Adam([tex_opt], lr=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: 0.1 ** (float(x) / float(MAX_ITER)))

    output_dir = create_out_dir("training_out", name)
    save_model(pos, pos_idx, vtx_uv, output_dir + "/")

    training_sample = random.sample(range(gnd_truth.shape[0]), int(gnd_truth.shape[0]*(3/4)))
    test_sample = list(set(range(gnd_truth.shape[0])) - set(training_sample))

    for i in range(MAX_ITER + 1):
        angle_index = training_sample[random.sample(range(len(training_sample)), 1)[0]]
        angle = 360 - (angle_index * 360 / gnd_truth.shape[0])

        if filter_planes:
            pos, pos_idx, vtx_uv, _ = merge_planes(planes, angle=angle, filter=True, filter_angle=filter_angle)

        # Render optimized model and fetch ground truth
        color_opt = render(pos, angle, cuda_ctx, pos_idx, vtx_uv, tex_opt, ranges, len(planes))
        color_gnd = gnd_truth[angle_index].cuda()

        loss = torch.mean((color_gnd - color_opt) ** 2)  # L2 pixel loss.
        if L1:
            loss = F.l1_loss(color_opt, color_gnd)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if i % SAVE_INTERVAL == 0:
            render_save_path = output_dir + "/renders/" + str(i) + ".png"
            save_comparison_image(color_opt, color_gnd, render_save_path)
            save_image(tex_opt, output_dir + "/textures/" + str(i))


    if evaluate:
        rmse_avg = []
        ssim_avg = []
        psnr_avg = []
        sre_avg = []

        for angle_index in test_sample:
            angle = 360 - (angle_index * 360 / gnd_truth.shape[0])

            color_opt = render(pos, angle, cuda_ctx, pos_idx, vtx_uv, tex_opt, ranges, len(planes))
            color_gnd = gnd_truth[angle_index].numpy()

            rmse_avg.append(qm.rmse(org_img=color_opt, pred_img=color_gnd))
            ssim_avg.append(qm.ssim(org_img=color_opt, pred_img=color_gnd))
            psnr_avg.append(qm.psnr(org_img=color_opt, pred_img=color_gnd))
            sre_avg.append(qm.sre(org_img=color_opt, pred_img=color_gnd))

        avg_results = [
            "rmse_avg: " + str(np.mean(rmse_avg)),
            "ssim_avg: " + str(np.mean(ssim_avg)),
            "psnr_avg: " + str(np.mean(psnr_avg)),
            "sre_avg: " + str(np.mean(sre_avg)),
        ]
        with open(output_dir + "/avg_results.txt", "w") as file:
            file.write("\n".join(avg_results))


if __name__ == '__main__':
    run(MAX_ITER=10000, SAVE_INTERVAL=100, plane_angles=[0, 45, 90, 135])
