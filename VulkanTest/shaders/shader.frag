#version 450

layout(location = 0) in vec3 fragPositionWS;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

layout(binding = 1) uniform sampler2D texSampler[5];

const float PI = 3.14159265359;
const vec3 cameraPos = vec3(2.0, 3.0, 1.0);
const vec3 lightDir = vec3(1.0, 1.0, -1.0);

float DistributionGGX(vec3 N, vec3 H, float roughness)
{
    float a = roughness * roughness;
    float asqr = a * a;
    float NdotH = max(dot(N, H), 0.0);
    float NdotH2 = NdotH * NdotH;

    float num = asqr;
    float denom = (NdotH2 * (asqr - 1.0) + 1.0);
    denom = PI * denom * denom;

    return num / denom;
}

float GeometrySchlickGGX(float NdotV, float roughness)
{
    float r = roughness + 1.0;
    float k = (r * r) / 8.0;

    float denom = NdotV * (1.0 - k) + k;
    return NdotV / denom;
}

float GeometrySmith(vec3 N, vec3 V, vec3 L, float roughness)
{
    float NdotV = max(dot(N, V), 0.0);
    float NdotL = max(dot(N, L), 0.0);
    float ggx1 = GeometrySchlickGGX(NdotV, roughness);
    float ggx2 = GeometrySchlickGGX(NdotL, roughness);
    return ggx1 * ggx2;
}

vec3 FresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
}

void main()
{
    vec3 albedo = texture(texSampler[0], fragTexCoord).rgb;
    vec3 normal = texture(texSampler[1], fragTexCoord).rgb * 2 - 1;
    float ao = texture(texSampler[2], fragTexCoord).r;
    float metallic = texture(texSampler[3], fragTexCoord).r;
    float roughness = texture(texSampler[4], fragTexCoord).r;

    vec3 N = normalize(normal);
    vec3 V = normalize(cameraPos - fragPositionWS);
    vec3 L = normalize(-lightDir);
    vec3 H = normalize(V + L);
    
    // Fresnel reflectance at normal incidence
    vec3 F0 = vec3(0.04);
    F0 = mix(F0, albedo, metallic);

    // Cook-Torrance BRDF
    float NDF = DistributionGGX(N, H, roughness);
    float G   = GeometrySmith(N, V, L, roughness);
    vec3  F   = FresnelSchlick(max(dot(H, V), 0.0), F0);

    vec3 nominator = NDF * G * F;
    float denom = 4.0 * max(dot(N, V), 0.0) * max(dot(N, L), 0.0) + 0.001;
    vec3 specular = nominator / denom;

    vec3 kS = F;
    vec3 kD = vec3(1.0) - kS;
    kD *= 1.0 - metallic;

    float NdotL = max(dot(N, L), 0.0);
    vec3 irradiance = vec3(1.0); // White directional light, no attenuation

    vec3 Lo = (kD * albedo / PI + specular) * irradiance * NdotL;

    // Final color (no ambient/IBL)
    vec3 ambient = vec3(0.05) * albedo * ao;
    vec3 color = ambient + Lo;

    // Tone mapping + gamma correction
    color = color / (color + vec3(1.0));
    //color = pow(color, vec3(1.0 / 2.2));

    outColor = vec4(color, 1.0);
}