non-fluents nf_USV_obstacle_nav_v3 {
    domain = USV_obstacle_nav_v3;

    objects{
      obs : {o1, o2, o3};
      USV : {b1};
    };
    non-fluents {
      //define goal coordinates
      XG = 100;
      YG = 100;
      //damage and colreg radii
      dmgRad = 7;
      crgRad = 15;
    };

}

instance USV_obstacle_nav_inst_v3 {
    domain = USV_obstacle_nav_v3;
    non-fluents = nf_USV_obstacle_nav_v3;
    init-state{

      //inital positioning of ASV
      xPos(b1) = 0.0;
      yPos(b1) = 0.0;

      //obstacle states
      obsXp(o1) = 40.0;
      obsYp(o1) = 40.0;
      obsV(o1) = 0.75;
      obsHead(o1) = -2.356194;

      obsXp(o2) = 60.0;
      obsYp(o2) = 30.0;
      obsV(o2) = 0.75;
      obsHead(o2) = 2.356194;

      obsXp(o3) = 70.0;
      obsYp(o3) = 50.0;
      obsV(o3) = 0.5;
      obsHead(o3) = 0.7853982;



    };
    //horizon = timesteps
    max-nondef-actions = 2;
    horizon = 120;
    discount = 1.0;
}
