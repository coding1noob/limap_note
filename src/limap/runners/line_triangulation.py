import os

import pycolmap
from pycolmap import logging
from tqdm import tqdm

import limap.base as base
import limap.merging as merging
import limap.optimize as optimize
import limap.pointsfm as pointsfm
import limap.runners as runners
import limap.triangulation as triangulation
import limap.util.io as limapio
import limap.visualize as limapvis
import limap.vplib as vplib


def line_triangulation(cfg, imagecols, neighbors=None, ranges=None):
    """
    Main interface of line triangulation over multi-view images.

    Args:
        cfg (dict): Configuration. \
            Refer to :file:`cfgs/triangulation/default.yaml` as an example
        imagecols (:class:`limap.base.ImageCollection`): \
            The image collection corresponding to all the images of interest
        neighbors (dict[int -> list[int]], optional): \
            visual neighbors for each image. By default we compute \
            neighbor information from the covisibility of COLMAP triangulation.
        ranges (pair of :class:`np.array` each of shape (3,), optional): \
            robust 3D ranges for the scene. By default we compute \
            range information from the COLMAP triangulation.
    Returns:
        list[:class:`limap.base.LineTrack`]: list of output 3D line tracks
    """
    logging.info(f"[LOG] Number of images: {imagecols.NumImages()}")
    cfg = runners.setup(cfg)
    # var2d 是2D线段的像素级噪声方差，用于三角化时的不确定性建模
    # yaml 里默认设为 -1，表示"自动按检测器选"。这里就是根据检测器名（deeplsd）去查表
    detector_name = cfg["line2d"]["detector"]["method"]
    if cfg["triangulation"]["var2d"] == -1:
        cfg["triangulation"]["var2d"] = cfg["var2d"][detector_name]
    # undistort images，去畸变
    if not imagecols.IsUndistorted():
        imagecols = runners.undistort_images(
            imagecols,
            os.path.join(cfg["dir_save"], cfg["undistortion_output_dir"]),
            skip_exists=cfg["load_undistort"] or cfg["skip_exists"],
            n_jobs=cfg["n_jobs"],
        )
    # resize cameras
    # 确保去畸变完成
    assert imagecols.IsUndistorted()
    # 把图像的长边限制在 max_image_dim=1600 像素以内。这里不是真正缩放图像文件，
    # 而是更新 imagecols 里的相机内参（焦距、主点等按比例缩放），让后续计算在这个分辨率下进行，以控制速度和内存
    if cfg["max_image_dim"] != -1 and cfg["max_image_dim"] is not None:
        imagecols.set_max_image_dim(cfg["max_image_dim"])
    # 保存图像列表和集合至 image_list.txt ，方便断点续跑
    limapio.save_txt_imname_dict(
        os.path.join(cfg["dir_save"], "image_list.txt"),
        imagecols.get_image_name_dict(),
    )
    limapio.save_npy(
        os.path.join(cfg["dir_save"], "imagecols.npy"), imagecols.as_dict()
    )

    ##########################################################
    # [A] sfm metainfos (neighbors, ranges)
    ##########################################################
    sfminfos_colmap_folder = None
    if neighbors is None:
    # 没有邻居信息 → 临时跑一次 COLMAP 点云重建来计算
        sfminfos_colmap_folder, neighbors, ranges = runners.compute_sfminfos(
            cfg, imagecols
        )
    # 一般都传邻域，走这条
    else:   
        limapio.save_txt_metainfos(
            os.path.join(cfg["dir_save"], "metainfos.txt"), neighbors, ranges
        )
        neighbors = imagecols.update_neighbors(neighbors)
        # 截断每张图的邻居数量，最多保留 n_neighbors=20 个
        for img_id, _ in neighbors.items():
            neighbors[img_id] = neighbors[img_id][: cfg["n_neighbors"]]
    # 最后保存邻居和范围信息到 metainfos.txt 里，方便断点续跑
    limapio.save_txt_metainfos(
        os.path.join(cfg["dir_save"], "metainfos.txt"), neighbors, ranges
    )

    ##########################################################
    # [B] get 2D line segments for each image
    ##########################################################

    # 先 判断是否需要描述子
    # use_exhaustive_matcher=False（默认）→ compute_descinfo = True
    # 逻辑是：用普通matcher（GlueStick）需要描述子才能匹配；用穷举matcher则直接两两比较几何关系，不需要描述子
    compute_descinfo = not cfg["triangulation"]["use_exhaustive_matcher"]
    # 如果已经有预计算的匹配结果（load_match=True）或已有检测结果（load_det=True），也不需要重新提取描述子
    compute_descinfo = (
        compute_descinfo and (not cfg["load_match"]) and (not cfg["load_det"])
    ) or cfg["line2d"]["compute_descinfo"]

    # 用 DeepLSD 检测2D线段
    # 用 Wireframe 提取每条线段的描述子，结果存入 line_detections/deeplsd/descinfos/wireframe
    all_2d_segs, descinfo_folder = runners.compute_2d_segs(
        cfg, imagecols, compute_descinfo=compute_descinfo
    )

    ##########################################################
    # [C] get line matches
    ##########################################################
    if not cfg["triangulation"]["use_exhaustive_matcher"]:
        matches_dir = runners.compute_matches(
            cfg, descinfo_folder, imagecols.get_img_ids(), neighbors
        )

    ##########################################################
    # [D] multi-view triangulation
    ##########################################################
    Triangulator = triangulation.GlobalLineTriangulator(cfg["triangulation"])
    Triangulator.SetRanges(ranges)
    all_2d_lines = base.get_all_lines_2d(all_2d_segs)
    Triangulator.Init(all_2d_lines, imagecols)
    if cfg["triangulation"]["use_vp"]:
        vpdetector = vplib.get_vp_detector(
            cfg["triangulation"]["vpdet_config"],
            n_jobs=cfg["triangulation"]["vpdet_config"]["n_jobs"],
        )
        vpresults = vpdetector.detect_vp_all_images(
            all_2d_lines, imagecols.get_map_camviews()
        )
        Triangulator.InitVPResults(vpresults)
    # get 2d bipartites from pointsfm model
    if cfg["triangulation"]["use_pointsfm"]["enable"]:
        if cfg["triangulation"]["use_pointsfm"]["colmap_folder"] is None:
            colmap_model_path = None
            # check if colmap model exists from sfminfos computation
            if (
                cfg["triangulation"]["use_pointsfm"]["reuse_sfminfos_colmap"]
                and sfminfos_colmap_folder is not None
            ):
                colmap_model_path = os.path.join(
                    sfminfos_colmap_folder, "sparse"
                )
                if not pointsfm.check_exists_colmap_model(colmap_model_path):
                    colmap_model_path = None
            # retriangulate
            if colmap_model_path is None:
                colmap_output_path = os.path.join(
                    cfg["dir_save"], "colmap_outputs_junctions"
                )
                input_neighbors = None
                if cfg["triangulation"]["use_pointsfm"]["use_neighbors"]:
                    input_neighbors = neighbors
                pointsfm.run_colmap_sfm_with_known_poses(
                    cfg["sfm"],
                    imagecols,
                    output_path=colmap_output_path,
                    skip_exists=cfg["skip_exists"],
                    neighbors=input_neighbors,
                )
                colmap_model_path = os.path.join(colmap_output_path, "sparse")
        else:
            colmap_model_path = cfg["triangulation"]["use_pointsfm"][
                "colmap_folder"
            ]
        reconstruction = pycolmap.Reconstruction(colmap_model_path)
        all_bpt2ds, sfm_points = runners.compute_2d_bipartites_from_colmap(
            reconstruction, imagecols, all_2d_lines, cfg["structures"]["bpt2d"]
        )
        Triangulator.SetBipartites2d(all_bpt2ds)
        if cfg["triangulation"]["use_pointsfm"]["use_triangulated_points"]:
            Triangulator.SetSfMPoints(sfm_points)
    # triangulate
    logging.info("Start multi-view triangulation...")
    for img_id in tqdm(imagecols.get_img_ids()):
        if cfg["triangulation"]["use_exhaustive_matcher"]:
            Triangulator.TriangulateImageExhaustiveMatch(
                img_id, neighbors[img_id]
            )
        else:
            matches = limapio.read_npy(
                os.path.join(matches_dir, f"matches_{img_id}.npy")
            ).item()
            Triangulator.TriangulateImage(img_id, matches)
    linetracks = Triangulator.ComputeLineTracks()

    # filtering 2d supports
    linetracks = merging.filter_tracks_by_reprojection(
        linetracks,
        imagecols,
        cfg["triangulation"]["filtering2d"]["th_angular_2d"],
        cfg["triangulation"]["filtering2d"]["th_perp_2d"],
    )
    if not cfg["triangulation"]["remerging"]["disable"]:
        # remerging
        linker3d = base.LineLinker3d(
            cfg["triangulation"]["remerging"]["linker3d"]
        )
        linetracks = merging.remerge(linker3d, linetracks)
        linetracks = merging.filter_tracks_by_reprojection(
            linetracks,
            imagecols,
            cfg["triangulation"]["filtering2d"]["th_angular_2d"],
            cfg["triangulation"]["filtering2d"]["th_perp_2d"],
        )
    linetracks = merging.filter_tracks_by_sensitivity(
        linetracks,
        imagecols,
        cfg["triangulation"]["filtering2d"]["th_sv_angular_3d"],
        cfg["triangulation"]["filtering2d"]["th_sv_num_supports"],
    )
    linetracks = merging.filter_tracks_by_overlap(
        linetracks,
        imagecols,
        cfg["triangulation"]["filtering2d"]["th_overlap"],
        cfg["triangulation"]["filtering2d"]["th_overlap_num_supports"],
    )
    validtracks = [
        track
        for track in linetracks
        if track.count_images() >= cfg["n_visible_views"]
    ]

    ##########################################################
    # [E] geometric refinement
    ##########################################################
    if not cfg["refinement"]["disable"]:
        cfg_ba = optimize.HybridBAConfig(cfg["refinement"])
        cfg_ba.set_constant_camera()
        ba_engine = optimize.solve_line_bundle_adjustment(
            cfg["refinement"], imagecols, linetracks, max_num_iterations=200
        )
        linetracks_map = ba_engine.GetOutputLineTracks(
            num_outliers=cfg["refinement"]["num_outliers_aggregator"]
        )
        linetracks = [track for (track_id, track) in linetracks_map.items()]

    ##########################################################
    # [F] output and visualization
    ##########################################################
    # save tracks
    limapio.save_txt_linetracks(
        os.path.join(cfg["dir_save"], "alltracks.txt"),
        linetracks,
        n_visible_views=4,
    )
    limapio.save_folder_linetracks_with_info(
        os.path.join(cfg["dir_save"], cfg["output_folder"]),
        linetracks,
        config=cfg,
        imagecols=imagecols,
        all_2d_segs=all_2d_segs,
    )
    VisTrack = limapvis.Open3DTrackVisualizer(linetracks)
    VisTrack.report()
    limapio.save_obj(
        os.path.join(
            cfg["dir_save"],
            "triangulated_lines_nv{}.obj".format(cfg["n_visible_views"]),
        ),
        VisTrack.get_lines_np(n_visible_views=cfg["n_visible_views"]),
    )

    # visualize
    if cfg["visualize"]:
        validtracks = [
            track
            for track in linetracks
            if track.count_images() >= cfg["n_visible_views"]
        ]

        def report_track(track_id):
            limapvis.visualize_line_track(
                imagecols,
                validtracks[track_id],
                prefix=f"track.{track_id}",
            )

        logging.info(
            "Visualization about to start. Please ensure you have a graphical "
            "backend (e.g., X11, Wayland, or a Jupyter display) available."
        )
        input(
            "Press Enter to continue with visualization, or Ctrl+C to abort..."
        )
        VisTrack.vis_reconstruction(
            imagecols, n_visible_views=cfg["n_visible_views"]
        )
    return linetracks
